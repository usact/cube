import math
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
from typing import Dict
import torch.nn as nn

# if torch.xpu doesn’t exist, insert a stub so is_available() returns False
if not hasattr(torch, "xpu"):
    class _XPU:
        @staticmethod
        def is_available():
            return False
    torch.xpu = _XPU

from diffusers import UNet2DConditionModel
# from diffusers.models.attention import Attention as Original2DAttention # <— true SD attention class  
try:
    # For recent versions of diffusers, the self-attn blocks are named CrossAttention
    from diffusers.models.attention import CrossAttention as Original2DAttention
except ImportError:
    from diffusers.models.attention import Attention   as Original2DAttention  # fallback for older versions :contentReference[oaicite:0]{index=0}


from cl.model.positional_encoding import CubemapPositionalEncoding
from cl.model.attention        import inflate_attention_layer
from cl.model.normalization    import replace_group_norms
from transformers import BitsAndBytesConfig


class CubeDiffModel(nn.Module):
    def __init__(
        self,
        # pretrained_model_name: str,
        base_unet: UNet2DConditionModel,
        num_faces: int = 6,
        uv_dim: int   = 10,
        skip_weight_copy: bool = False, # When U-Net is 4-bit, skip copying pretrained weights into the inflated layers
                                        # (so avoid all dequantize/view_as hacks). djx: 2025-06-5, if skip copy is true, the panorama will be random noisy colors even trained for many data samples
        config: dict = None,  # optional config dict to pass to inflate_attention_layer
    ):
        """
        base_unet: a pre-loaded HF UNet2DConditionModel (cache-first).
        skip_weight_copy: if True, new inflated-attn weights are randomly init.
        """
        super().__init__()
        
        # — 1) Load only the U-Net in BF16
        self.skip_weight_copy = skip_weight_copy
        self.base_unet = base_unet 
        # — 2) Move the U-Net onto CUDA and record device/dtype
        self.model_dtype = torch.bfloat16
        self.base_unet = self.base_unet.to("cuda", dtype=self.model_dtype)
        self.device    = next(self.base_unet.parameters()).device

        # inflation -----------------------------------------------------------
        print("architecture.py - CubeDiffModel - 🔍 All Attention modules in UNet:")
        for name, module in self.base_unet.named_modules():
            if isinstance(module, Original2DAttention):
                # print("   ", name, "(cross)" if module.is_cross_attention else "(self)")
                # With this safe version:
                print("   ", name, "(cross)" if hasattr(module, 'is_cross_attention') and module.is_cross_attention else "(self)")
                
        # — 3) Inflate Stable Diffusion’s (only the UNet’s self‐attention) layers → cross‐face Attention  
        print("architecture.py - CubeDiffModel - DEBUG: All trainable param names in base_unet:")
        for name, p in self.base_unet.named_parameters():
            print(f"architecture.py - CubeDiffModel - base_unet name is {name}")
        
        self._inflated_modules = [] 
        self._inflated_module_names = []

        # inflating all self anc cross attention layers:
        for name, module in list(self.base_unet.named_modules()):
            # 1) catch every 2D attention layer (self + cross)
            if not isinstance(module, Original2DAttention):
                continue

            # 2) perform inflation exactly as before
            print(f"architecture.py - CubeDiffModel - Inflating {name}")
            parent = self.base_unet
            parts  = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)

            inflated_layer = inflate_attention_layer(
                original_attn=module,
                skip_copy=skip_weight_copy
            )
            self._inflated_module_names.append(name)  # keep track of inflated module names
            self._inflated_modules.append(inflated_layer)
            setattr(parent, parts[-1], inflated_layer)

            print(f"architecture.py - CubeDiffModel - Replaced base_unet {name} with inflated_layer")
        
        # ---- INSERT ASSERTS HERE: ENSURE INFLATION TOOK PLACE ----------------------
        # Updated check: Iterate through modules and find those with the `is_inflated` flag.
        # This is more robust than checking parameter names.
        total_inflated_params = 0
        for module in self.base_unet.modules():
            if hasattr(module, "is_inflated") and module.is_inflated:
                # Sum the parameters of this specific inflated module
                total_inflated_params += sum(p.numel() for p in module.parameters())         

        assert total_inflated_params > 0, (
            "❌ No inflated-attention parameters found. "
            "Did you import and replace Original2DAttention correctly?"
        )
        print(f"architecture.py - CubeDiffModel - ___init__ - [DEBUG] total_inflated_params: {total_inflated_params} ({total_inflated_params/(10**6):.4f} million)")

        for name in self._inflated_module_names:
            print(f"architecture.py - CubeDiffModel - Inflated layer module: {name}")

          
        # freeze all unet prameters 
        for p in self.base_unet.parameters():
            p.requires_grad = False
        
        # Unfreeze inflated-attn parameters only
        for mod in self._inflated_modules:
            for p in mod.parameters():
                p.requires_grad = True   

        # Immediately after inflation, for debugging only:
        print("\n=== Inflated-Attention Layers ===")
        for name, module in self.base_unet.named_modules():
            if hasattr(module, "is_inflated") and module.is_inflated:
                print(f"archtecture.py - __init__  - unet -  ✅ {name}: {module}")
        print("=== End inflated list ===\n")
        # inflation end -----------------------------------------------------------

        # applied SynchronizedGroupNorm for U-net

        print("architecture.py - CubeDiffModel - Replacing GroupNorms in UNet for color consistency")
        num_replaced = replace_group_norms(self.base_unet, in_place=True)
        print(f"architecture.py - CubeDiffModel - init - Replaced {num_replaced} GroupNorm layers with SGN in UNet")

        # — 5) Positional encoding (UV channels) -------------------------------
        self.positional_encoding = CubemapPositionalEncoding(
            num_faces=num_faces,
            embedding_dim=uv_dim
        ).to(self.device)  # ensure it lives on GPU as well

        # — 6) Patch conv_in to accept [latent(4) + mask(1) + uv_dim] channels
        old_conv = self.base_unet.conv_in
        in_ch    = old_conv.in_channels             # typically 4
        mask_ch  = 1
        out_ch   = old_conv.out_channels            # typically 320
        k, s, p  = old_conv.kernel_size, old_conv.stride, old_conv.padding

        new_conv = nn.Conv2d(
            in_channels=in_ch + mask_ch + uv_dim, # in_ch is latent_channels, 4 + 1 + 10 = 15
            out_channels=out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=(old_conv.bias is not None),
        )
        
        # copy the old latent weights & biases, zero‐init new channels
        # if no copy, the panorama will be noisy colors even trained for 12k+ data samples
        with torch.no_grad():
            # 1) copy original latent→feature weights
            new_conv.weight[:, :in_ch].copy_(self.base_unet.conv_in.weight)
            # 2) zero-init UV & mask weights so model can learn from scratch
            new_conv.weight[:, in_ch:].zero_()
            new_conv.bias.copy_(self.base_unet.conv_in.bias)
        
        # move the new conv to the same device/dtype
        new_conv = new_conv.to(device=self.device, dtype=self.model_dtype)
        self.base_unet.conv_in = new_conv # copy into the U-Net
        # ── now freeze it exactly like the rest of the U-Net ──
        # This guarantees that conv_in remains exactly the pretrained mapping on all 64×64 latents during the first few thousand warmup steps
        # so we don’t blow up the very first denoising layer.
        for p in self.base_unet.conv_in.parameters():
            p.requires_grad = False

        # — 7) Enable gradient checkpointing on the U-Net
        self.base_unet.enable_gradient_checkpointing()

        # — 9) Face-ID & spherical embeddings
        self.face_emb = nn.Embedding(num_faces, uv_dim).to(self.device, self.model_dtype)
        self.sph_emb  = nn.Sequential(
            nn.Linear(2, uv_dim),
            nn.SiLU(),
            nn.Linear(uv_dim, uv_dim),
        ).to(self.device, self.model_dtype)

    
    def forward(
        self,
        latents: torch.Tensor,               # [B, F, C0, H, W] or [B*F, C0, H, W]
        timesteps: torch.Tensor,             # [B] or [B*F]
        encoder_hidden_states: torch.Tensor, # [B, D] | [B, L, D] | [B*F, D] | [B*F, L, D]
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # 0) Normalize to 4D “flat” batch for positional_encoding
        if latents.ndim == 5:
            B, F, C0, H, W = latents.shape
            lat_flat = latents.view(B * F, C0, H, W)  # 5D → 4D
        else:  # [B*F, C0, H, W]
            Bf, C0, H, W = latents.shape
            F = getattr(self, "num_faces", 6)
            B = Bf // F
            lat_flat = latents

        # now that lat_flat is 4D, convert to channels_last NHWC
        # This unlocks maximum Tensor‐Core utilization (≈ 20–30 % faster) on Ampere GPUs
        lat_flat = lat_flat.contiguous(memory_format=torch.channels_last) # NOW apply channels_last to 4D tensor

        # 1) Cast all to correct device & dtype
        lat_flat = lat_flat.to(self.device, self.model_dtype)
        timesteps = timesteps.to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.device, self.model_dtype)

        # 2) Positional encoding on 4D latents → [B*F, E_total, H, W]
        pos_all = self.positional_encoding(lat_flat)

        # 3) Determine how many pos channels we actually need
        expected_in = self.base_unet.conv_in.weight.shape[1]  # e.g. 15
        pos_channels_needed = expected_in - C0 - 1  # subtract latent + mask

        # 4) Split and slice positional channels to match conv_in
        latent_part = lat_flat                          # [B*F, C0, H, W]
        mask        = torch.ones((latent_part.shape[0], 1, H, W),
                                    device=self.device, dtype=self.model_dtype)
        pos_part    = pos_all[:, :pos_channels_needed, :, :]  # [B*F, pos_channels_needed, H, W]

        # 5) Concatenate into the exact channels the U-Net expects
        #    → [B*F, C0 + 1 + pos_channels_needed, H, W]
        lat_in = torch.cat([latent_part, mask, pos_part], dim=1)

        # 6) Tile timesteps to [B*F] if they were [B]
        if timesteps.ndim == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.repeat_interleave(F)

        # 7) Tile encoder_hidden_states similarly to [B*F,...]
        N = encoder_hidden_states.shape[0]
        if N == 1 and B > 1:
            # Special case: single embedding needs to be broadcast to all B batches first
            if encoder_hidden_states.ndim == 2:
                # [1, D] → [B, D] → [B*F, D]
                encoder_hidden_states = encoder_hidden_states.expand(B, -1)
                encoder_hidden_states = (
                    encoder_hidden_states.unsqueeze(1)
                                        .expand(-1, F, -1)
                                        .reshape(B * F, -1)
                )
            elif encoder_hidden_states.ndim == 3:
                # [1, L, D] → [B, L, D] → [B*F, L, D]
                L, D = encoder_hidden_states.shape[1:]
                encoder_hidden_states = encoder_hidden_states.expand(B, -1, -1)
                encoder_hidden_states = (
                    encoder_hidden_states.unsqueeze(1)
                                        .expand(-1, F, -1, -1)
                                        .reshape(B * F, L, D)
                )
            else:
                raise ValueError(f"Unsupported encoder_hidden_states ndim={encoder_hidden_states.ndim}")
            
        elif N == B:
            if encoder_hidden_states.ndim == 2:
                encoder_hidden_states = (
                    encoder_hidden_states.unsqueeze(1)
                                        .expand(-1, F, -1)
                                        .reshape(B * F, -1)
                )
            elif encoder_hidden_states.ndim == 3:
                L, D = encoder_hidden_states.shape[1:]
                encoder_hidden_states = (
                    encoder_hidden_states.unsqueeze(1)
                                        .expand(-1, F, -1, -1)
                                        .reshape(B * F, L, D)
                )
            else:
                raise ValueError(f"Unsupported encoder_hidden_states ndim={encoder_hidden_states.ndim}")
        elif N != B * F:
            # This is where the error was occurring
            print(f"architecture.py - forward - ERROR DEBUG: N={N}, B={B}, F={F}, B*F={B*F}")
            print(f"architecture.py - forward - encoder_hidden_states.shape: {encoder_hidden_states.shape}")
            print(f"architecture.py - forward - latents.shape: {latents.shape}")
            raise ValueError(f"Wrong embedding batch: got {N}, expected 1, {B} or {B*F}")

        # 8) Forward through U-Net under autocast
        with torch.autocast("cuda", dtype=self.model_dtype):
            unet_out = self.base_unet(
                sample=lat_in,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            ).sample

        # 9) Un-flatten back to [B, F, C_out, H, W]
        C_out = unet_out.shape[1]
        unet_out = unet_out.view(B, F, C_out, H, W)
        return {"sample": unet_out}

