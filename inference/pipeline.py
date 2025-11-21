# cl/model/pipeline.py

import torch
import numpy as np
from typing import List
from diffusers import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler # DDIMScheduler # DPMSolverMultistepScheduler # EulerAncestralDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from data.preprocessing import cubemap_to_equirect
from model.architecture import CubeDiffModel # contains inflated‐attention U-Net + SyncBN
from model.normalization import replace_group_norms

# Face adjacency map for a standard cubemap
_FACE_ADJ = {
    0: {"left": 3, "right": 1, "top": 4, "bottom": 5},  # front
    1: {"left": 0, "right": 2, "top": 4, "bottom": 5},  # right
    2: {"left": 1, "right": 3, "top": 4, "bottom": 5},  # back
    3: {"left": 2, "right": 0, "top": 4, "bottom": 5},  # left
    4: {"left": 3, "right": 1, "top": 2, "bottom": 0},  # top
    5: {"left": 3, "right": 1, "top": 0, "bottom": 2},  # bottom
}

def get_adjacent_faces(face_idx: int):
    """
    Returns (left, right, top, bottom) face indices for face_idx.
    """
    d = _FACE_ADJ[face_idx]
    return d["left"], d["right"], d["top"], d["bottom"]


class CubeDiffPipeline:
    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        use_safetensors: bool = True,
        dtype: torch.dtype = None,
        height: int = 512,
        width: int = 512,
        config=None,  # CubeDiffModel config
    ):
        self.device = torch.device(device)
        dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)

        # 1) Safe loader that tries local cache first, then downloads:
        def safe_load(cls, repo_id, **kwargs):
            try:
                return cls.from_pretrained(repo_id, local_files_only=True, **kwargs)
            except (OSError, ValueError, TypeError):
                return cls.from_pretrained(repo_id, **kwargs)

        # 2) Load the five core modules, each with cache‐first:
        print("Loading VAE…")
        vae = safe_load(
            AutoencoderKL,
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
        ).to(self.device).eval()

        # 1) Synchronize GroupNorm in the VAE (sec 4.2 of CubeDiff) 
        replaced = replace_group_norms(vae, in_place=True)
        if replaced == 0:
            print(f"pipeline.py - CubeDiffPipeline - __init__ - ⚠️ replace_group_norms: no GroupNorm found in VAE—skipping sync step.")
        else:
            print(f"pipeline.py - CubeDiffPipeline - __init__ - replace_group_norms: replaced {replaced} GroupNorm layers with SynchronizedGroupNorm.")
        assert replaced > 0, "❌ No GroupNorm found in VAE to synchronize—check your model version."
        
        # 2) Freeze all VAE params so only the UNet is trained
        for p in vae.parameters():
            p.requires_grad = False

        print("pipeline.py - CubeDiffPipeline - safe loading U-Net…")
        # inflated, norm-replaced, and then used at inference.
        hf_unet = safe_load(
            UNet2DConditionModel,
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
            load_in_4bit=False  # ← must be false!, Your base U-Net must be loaded in a precision that supports weight copy
                                # your inflate_attention_layer will bail out of copying (“can’t view_as() that tensor shape”) 
                                # and leave the 3D kernels random
        ).to(self.device)

        # 3) Wrap & inflate the U-Net
        print("pipelien.py - CubeDiffPipeline - Wrapping & inflating U-Net (CubeDiffModel)…")
        
        # 2) Wrap it:
        cube_model = CubeDiffModel(
            base_unet=hf_unet,
            skip_weight_copy=False, 
            config=config,
        )
        
        # 4) Load tokenizer + text encoder
        print("Loading CLIP tokenizer & text encoder…")
        tokenizer = safe_load(
            CLIPTokenizer,
            "openai/clip-vit-large-patch14",
            use_fast=True
        )
        print("Loading CLIP tokenizer done")
        text_encoder = safe_load(
            CLIPTextModel,
            "openai/clip-vit-large-patch14",
            torch_dtype=dtype,
            use_safetensors=use_safetensors,
        ).to(self.device).eval()
        print("Loading CLIP text encode done")

        # 5) Load a velocity‐prediction scheduler (for v-prediction sampling)
        # print("Loading Karras ancestral scheduler (v-prediction)…")
        
        # ── revert to ε-prediction DDIM ──
    #     scheduler = DDIMScheduler.from_pretrained(
    #         pretrained_model_name,
    #         subfolder="scheduler",
    #         prediction_type="epsilon",      # ← explicit ε-prediction
    #         clip_sample=False,               # keep raw model output
    #    )
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler",
            prediction_type="epsilon",      # ← explicit ε-prediction
            clip_sample=False,               # keep raw model output
        )
        
        orig_get_variance = self.scheduler._get_variance

        def _get_variance_tensor(self, timestep, predicted_variance=None):
            # call original
            var = orig_get_variance(timestep, predicted_variance)
            # if it’s not a Tensor, wrap it
            if not torch.is_tensor(var):
                var = torch.tensor(var, device=self.device, dtype=self.model_dtype)
            # now clamp
            return torch.clamp(var, min=1e-20)

        # bind our patched method
        self.scheduler._get_variance = _get_variance_tensor.__get__(self.scheduler, type(self.scheduler))

        # Ensure alphas_cumprod and sigmas are NumPy on CPU for set_timesteps()
        # if hasattr(scheduler, "alphas_cumprod") and torch.is_tensor(scheduler.alphas_cumprod):
        #     scheduler.alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
        # if hasattr(scheduler, "sigmas") and torch.is_tensor(scheduler.sigmas):
        #     scheduler.sigmas = scheduler.sigmas.cpu().numpy()
        # print("Loading Karras ancestral scheduler (v-prediction) done")

        print("Loading Scheduler done")
        # 6) Store everything
        self.vae = vae
        self.model = cube_model # unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        # self.scheduler = scheduler
        self.height = height
        self.width = width
        self.latent_channels = 4 # VAE latent channels, e.g. 4 for SD1.5
        self.overlap = 5  # overlap between cubemap faces, e.g. 5 pixels

        # ── ensure scheduler alphas/betas live on the same device as the model ──
        for attr in ("timesteps", "alphas_cumprod", "alphas_cumprod_prev", "betas", "one"):
            val = getattr(self.scheduler, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(self.scheduler, attr, val.to(self.device))

        # ── robustly infer the VAE’s latent spatial dims ──
        with torch.no_grad():
            # make a dummy image at the target resolution,
            # in the _same_ dtype as the VAE’s weights:
            vae_dtype = next(self.vae.parameters()).dtype
            # make a dummy image at the target resolution
            dummy = torch.zeros(
                1,
                self.vae.config.in_channels,
                self.height,
                self.width,
                device=self.device,
                dtype=vae_dtype,
            )
            # encode it and sample -- diffusers’ AutoencoderKL returns a
            # LatentDistribution object with .sample() giving [1, C, H_lat, W_lat]
            latents = self.vae.encode(dummy).latent_dist.sample()
        # grab the true H_lat/W_lat from that tensor
        self.H_lat, self.W_lat = latents.shape[-2:]

        print("✅ CubeDiffPipeline initialized.")


    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        device: torch.device = torch.device("cuda"),
    ) -> np.ndarray:
        """
        Generate a single equirectangular panorama ([H*2, W*4, 3]) from text,
        ensuring each cubemap face remains in its fixed slot and orientation.

        We use classifier-free guidance with strict batch-shape consistency:
        - Build a 2× batch ([uncond; cond]) for the UNet.
        - Pass the same 2× batch into the scheduler.
        - Split the scheduler output and keep the first half, preserving face ordering.
        """
        # 1) Text encoding for classifier-free guidance
        toks = self.tokenizer(
            [""], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        uncond_emb = self.text_encoder(toks.input_ids)[0]      # [1, seq_len, D]
        toks = self.tokenizer(
            [prompt], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        cond_emb = self.text_encoder(toks.input_ids)[0]        # [1, seq_len, D]
        text_emb = torch.cat([uncond_emb, cond_emb], dim=0)    # [2, seq_len, D]

        # 2) Initialize cubemap latents
        B, F, C = 1, 6, self.latent_channels
        H_lat, W_lat = self.H_lat, self.W_lat
        sigma = getattr(self.scheduler, "init_noise_sigma", None) or self.scheduler.sigmas[0]
        latents = torch.randn((B, F, C, H_lat, W_lat), device=device) * sigma

        # 3) Scheduler setup (DDIM)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        overlap = self.overlap
        padded_h = H_lat + 2*overlap
        padded_w = W_lat + 2*overlap

        # 4) Denoising loop with proper guidance
        for t in self.scheduler.timesteps:
            # a) Pad for cross-face context
            tmp = torch.zeros((B, F, C, padded_h, padded_w), device=device)
            tmp[:, :, :, overlap:overlap+H_lat, overlap:overlap+W_lat] = latents
            for f in range(F):
                l, r, u, d = get_adjacent_faces(f)
                tmp[:, f, :, :overlap, overlap:overlap+W_lat]        = latents[:, u, :, H_lat-overlap:, :]
                tmp[:, f, :, overlap+H_lat:, overlap:overlap+W_lat]   = latents[:, d, :, :overlap,       :]
                tmp[:, f, :, overlap:overlap+H_lat, :overlap]         = latents[:, l, :, :, W_lat-overlap:]
                tmp[:, f, :, overlap:overlap+H_lat, overlap+W_lat:]   = latents[:, r, :, :, :overlap]
            # b) Flatten to [B*F, C, H, W]
            flat = tmp.view(B*F, C, padded_h, padded_w)

            # c) Build 2× batch for UNet: [uncond_latents; cond_latents]
            uncond, cond = flat, flat
            model_in = torch.cat([uncond, cond], dim=0)            # [2·B·F, C, H, W]

            # d) Prepare timesteps & embeddings for 2× batch
            t_in = torch.full((2*B*F,), int(t), device=device, dtype=torch.long)
            emb_un = text_emb[0:1].expand(B*F, -1, -1)
            emb_co = text_emb[1:2].expand(B*F, -1, -1)
            emb_all = torch.cat([emb_un, emb_co], dim=0)           # [2·B·F, seq_len, D]

            # e) UNet forward
            noise_pred = self.model(model_in, t_in, emb_all)["sample"]  # [2·B·F,...]
            u, c = noise_pred.chunk(2, dim=0)                              # each [B·F,...]

            # f) Classifier-free guidance
            guided = u + guidance_scale * (c - u)                         # [B·F,...]

            # g) Scheduler step: feed same 2× batch back in
            guided2 = torch.cat([guided, guided], dim=0)                 # [2·B·F,...]
            prev2 = self.scheduler.step(guided2, t, flat)["prev_sample"]

            # h) Extract first half to preserve ordering
            next_flat = prev2.chunk(2, dim=0)[0]                          # [B·F, C, H, W]
            latents = next_flat.view(B, F, C, padded_h, padded_w)[
                :, :, :, overlap:overlap+H_lat, overlap:overlap+W_lat
            ]

        # 5) Decode & rotate faces statically
        faces = []
        # Static rotations (in 90° CCW steps) to align each face to cubemap axes:
        # f index: 0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom
        # front:  0°  → 0 steps CCW
        # right: +90° → 3 steps CCW
        # back:  180° → 2 steps CCW
        # left:  -90° → 1 step  CCW
        # top/bottom aligned, no rotation
                # Static rotations (in 90° CCW steps) to align each face:
        # f index: 0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom
        # front:  0°  → 0 steps CCW
        # right: +90° → 3 steps CCW
        # back:   0°  → 0 steps (no rotation)
        # left:  180° → 2 steps CCW
        # top/bottom: no rotation
        rotation_map = {
            0: 0,  # front
            1: 3,  # right (90° CW)
            2: 0,  # back (0 flip) # from 90° flip
            3: 0,  # left (0° flip) # from 180 flip
            4: 0,  # top
            5: 0,  # bottom
        }  # index→90°-CCW steps
        for f in range(F):
            z = latents[0, f]
            dec = self.vae.decode(z.unsqueeze(0)/0.18215).sample[0]
            dec = (dec.clamp(-1,1) + 1)/2
            k = rotation_map.get(f, 0)
            if k:
                dec = torch.rot90(dec, k=k, dims=[1,2])
                print(f"pipeline.py - CubeDiffPipeline - generate - Rotated face {f} by {k*90}° CCW, dec shape is {dec.shape}, dec dtype is {dec.dtype}")
            faces.append(dec.cpu().float())

        # 6) Project via the unchanged cubemap_to_equirect
        print("pipeline.py - CubeDiffPipeline - generate - stacking faces and projecting to equirectangular...")
        print(f"pipeline.py - CubeDiffPipeline - generate - each face shape: {faces[0].shape}, num faces: {len(faces)}")
        cube_np = torch.stack(faces, 0).numpy()
        print(f"pipeline.py - generate - cube_np shape: {cube_np.shape}, dtype: {cube_np.dtype}, min: {cube_np.min()}, max: {cube_np.max()}")
        
        # in order to make cube_np (all faces) compatiable with "faces: Tensor[6, H, W, 3]" in preprocessing.cubemap_to_equirect()
        # permute faces ([6, 3, 512, 512) to be [6, 512, 512, 3]
        # convert from [6, 3, H, W] → [6, H, W, 3]
        cube_np = cube_np.transpose(0, 2, 3, 1)
        print(f"pipeline.py - generate - after cube_np.permute(0, 2, 3, 1), cube_np shape: {cube_np.shape}, dtype: {cube_np.dtype}, min: {cube_np.min()}, max: {cube_np.max()}")
        
        # pano = cubemap_to_equirect(cube_np, self.height*2, self.width*4)
        print(f"pipeline.py - generate - self.height: {self.height}, self.width: {self.width}")
        pano = cubemap_to_equirect(cube_np, self.height, self.width*2)
        print(f"pipeline.py - CubeDiffPipeline - generate - pano shape: {pano.shape}")
        # return pano.permute(1,2,0)  # [He,We,3]
        # return pano.permute(1, 2, 0).contiguous()  # [He, We, 3]
        return pano
