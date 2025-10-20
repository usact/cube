# File: cl/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention  # type: ignore

class InflatedAttention(nn.Module):
    """
    Inflated multi-head attention for cubemap faces.
    Takes input [B, N, C] where N = 6 * num_patches_per_face,
    splits N→(6 faces × L patches), attends across all faces,
    and returns [B, N, C].
    """
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0, context_dim=None):
        super().__init__()
        self.is_inflated = True
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        # Q/K/V projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        # use context_dim (text-embedding dim) for cross-attn keys/values
        cd = context_dim or query_dim
        self.to_k = nn.Linear(cd, inner_dim, bias=False)
        self.to_v = nn.Linear(cd, inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,              # [B×6, L, C]
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.BoolTensor    = None,
        **kwargs
    ) -> torch.FloatTensor:
        # 1) Recover original batch B and 6 faces
        total, L, C = hidden_states.shape
        FACES = 6
        if total % FACES != 0:
            raise ValueError(f"Batch size {total} not divisible by {FACES} faces")
        B = total // FACES

        # 2) Reshape into (B,6,L,C)
        # x = hidden_states.view(B, FACES, L, C)
        # if encoder_hidden_states is not None:
        #     ctx = encoder_hidden_states.view(B, FACES, L, C)
        # else:
        #     ctx = x
        
        # Reshape for cross-face attention
        # x = hidden_states.view(B, FACES * L, C)

        # 2) Reshape queries from [B*6, L, C] to [B, 6*L, C]
        # This groups all face tokens for a single panorama into one sequence.
        x_reshaped = hidden_states.view(B, FACES * L, C)


        # 3) Prepare context: if encoder_hidden_states given, reshape similarly (but allowing different seq length)
        # if encoder_hidden_states is not None:
        #     # total_ctx, M, Cctx = encoder_hidden_states.shape
        #     # if total_ctx % FACES != 0:
        #         # raise ValueError(f"Encoder batch size {total_ctx} not divisible by {FACES} faces")
        #     # ctx = encoder_hidden_states.view(B, FACES, M, Cctx).reshape(B, FACES * M, Cctx)
        #     # Cross-attention: context comes from encoder
        #     ctx = encoder_hidden_states
        # else:
        #     # ctx = x.reshape(B, FACES * L, C)
        #     # Self-attention: context is the input itself
        #     ctx = x

        # 3) Prepare the context (for keys and values)
        if encoder_hidden_states is not None:
            # CROSS-ATTENTION: Reshape context from [B*6, M, D] to [B, 6*M, D]
            # This makes the text embedding context compatible with the reshaped query.
            M, D = encoder_hidden_states.shape[1], encoder_hidden_states.shape[2]
            # ctx_reshaped = encoder_hidden_states.view(B, FACES * M, D)
            ctx_reshaped = encoder_hidden_states.reshape(B, FACES * M, D)
        else:
            # SELF-ATTENTION: The context is the input itself.
            ctx_reshaped = x_reshaped

        # 3) Merge faces back into tokens: [B, 6×L, C]
        # x   = x.reshape(B, FACES * L, C)
        # ctx = ctx.reshape(B, FACES * L, C)
        
        # 3) Merge faces back into tokens for queries: [B, 6×L, C]
        # x = x.reshape(B, FACES * L, C)
        # 'ctx' has already been prepared above, slicing handled later if needed

        # 4) Standard multi-head attention on [B, N=6L, C]
        # replace the explicit q @ k^T → softmax → (attn @ v) with PyTorch’s fused, block-wise attention. 
        # This never materializes the full [B, H, N, N] tensor on GPU.
        # 4) Memory-efficient scaled dot-product attention
        # q = self.to_q(x).view(B, -1, self.heads, C//self.heads).permute(0,2,1,3)
        # k = self.to_k(ctx).view(B, -1, self.heads, C//self.heads).permute(0,2,1,3)
        # v = self.to_v(ctx).view(B, -1, self.heads, C//self.heads).permute(0,2,1,3)

        # 4) Project queries, keys, and values from the reshaped tensors
        q = self.to_q(x_reshaped)
        k = self.to_k(ctx_reshaped)
        v = self.to_v(ctx_reshaped)

        # 5) Reshape for multi-head attention and perform scaled_dot_product_attention
        # This core mathematical operation is unchanged.
        q = q.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, -1, self.heads, C // self.heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        # PyTorch 2.0+ fused attention — blocks internally, avoids OOM
        
        # 6) Reshape output back to the original format [B, 6*L, C]
        out = out.transpose(1, 2).contiguous().view(B, FACES * L, C)
        out = self.to_out(out)
        return out.view(B * FACES, L, C)


def inflate_attention_layer(
    original_attn: Attention,
    skip_copy: bool = False
) -> InflatedAttention:
    """
    Replace a HuggingFace SD Attention with our InflatedAttention.
    When skip_copy=False, dequantize() and copy the 320×320 on-face weights
    into the InflatedAttention’s to_q/to_k/to_v/to_out[0] for the diagonal blocks.
    When skip_copy is True, leave them at random init (fast 4-bit path).
    """
    if not skip_copy:
        # DEBUG: confirm we’re copying pretrained conv‐in filters
        q_w_shape = tuple(original_attn.to_q.weight.shape)
        print(f"attention.py - skip_copy is {skip_copy} - ✅ Inflating attention @ {original_attn.__class__.__name__},\
               copying pretrained q‐weight shape {q_w_shape}")
    # Build the inflated module
    query_dim = original_attn.to_q.in_features
    dropout   = getattr(original_attn.to_out[1], "p", 0.0)
    heads     = original_attn.heads
    dim_head  = query_dim // heads
    
    # ctx_dim = original_attn.to_k.in_features if original_attn.is_cross_attention else query_dim
    # SAFE: Detect cross-attention by comparing dimensions
    # For runwayml/stable-diffusion-v1-5:
    # Self-attention: query_dim = ctx_dim = 320/640/1280 (spatial features)
    # Cross-attention: query_dim = 320/640/1280, ctx_dim = 768 (CLIP text features)
    k_input_dim = original_attn.to_k.in_features
    
    # If to_k input dimension differs from to_q, it's cross-attention
    if k_input_dim != query_dim:
        # Cross-attention: keys/values come from different space (text encoder)
        ctx_dim = k_input_dim
        attn_type = "cross-attention"
    else:
        # Self-attention: keys/values same space as queries
        ctx_dim = query_dim
        attn_type = "self-attention"
    
    print(f"attention.py - Detected {attn_type}: query_dim={query_dim}, ctx_dim={ctx_dim}")

    inflated = InflatedAttention(
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
        context_dim=ctx_dim
    )
    
    # replicate the pretrained weight across all blocks (diagonal and off-diagonal) so that initially there is no randomness
    # then fine-tune to carve out the correct cross-face interactions.
    if not skip_copy:
        # helper to extract the full [320×320] weight for both FP and 4-bit layers
        def get_weight(module: nn.Module):
            if hasattr(module, "dequantize"):
                return module.dequantize()
            if hasattr(module, "weight"):
                return module.weight.data
            if hasattr(module, "qweight"):
                return module.qweight.data
            raise RuntimeError(f"No weight found in {module}")

        # Extract and copy
        wq = get_weight(original_attn.to_q)
        wk = get_weight(original_attn.to_k)
        wv = get_weight(original_attn.to_v)
        wo = get_weight(original_attn.to_out[0])

        # Number of faces T=6 → full size is [T*C × T*C]
        C = wq.shape[0]
        T = inflated.to_q.weight.shape[0] // C
        
        # Block‐diagonal: each face gets its own copy, off‐diagonals zero
        # (torch.kron with identity builds exactly that)
        I = torch.eye(T, device=wq.device, dtype=wq.dtype)
        full_wq = torch.kron(I, wq)          # → [T·C, T·C]

        # Tile the pretrained weights across all T×T blocks
        # full_wq = wq.repeat(T, T)        # → [T*C, T*C]
        full_wk = wk.repeat(T, T)
        full_wv = wv.repeat(T, T)
        full_wo = wo.repeat(T, T)

        # Transpose if shapes mismatch
        # if inflated.to_q.weight.shape != wq.shape:
            # wq = wq.t().view_as(inflated.to_q.weight)

        # inflated.to_q.weight.data.copy_(wq)
        # inflated.to_k.weight.data.copy_(wk)
        # inflated.to_v.weight.data.copy_(wv)
        # inflated.to_out[0].weight.data.copy_(wo)

        # Copy into inflated parameters
        inflated.to_q.weight.data.copy_(full_wq)
        inflated.to_k.weight.data.copy_(full_wk)
        inflated.to_v.weight.data.copy_(full_wv)
        inflated.to_out[0].weight.data.copy_(full_wo)

        # Copy bias if present
        b = getattr(original_attn.to_out[0], "bias", None)
        if b is not None:
            inflated.to_out[0].bias.data.copy_(b.data)

    # ✅ Wrap with identifiable module name
    # Wrap InflatedAttention in an outer nn.Module and register it with the submodule name "inflated_attn" to satisfy CubeDiff's assertion.
    # The CubeDiffModel assertion checks:
    #     if "inflated_attn" in named_parameters() yields names like "inflated_attn.to_q.weight" only if the inflated attention is registered under that name.
    #     This approach matches SOTA practices in parameter injection and modular parameter-efficient fine-tuning (PEFT), e.g., LoRA, QLoRA, DreamBooth.

    # wrapper = nn.Module()
    # wrapper.inflated_attn = inflated

    # def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
    #     return wrapper.inflated_attn(hidden_states, encoder_hidden_states, attention_mask, **kwargs)

    # wrapper.forward = forward
    # return wrapper

    return inflated