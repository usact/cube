import torch
import torch.nn as nn
import torch.nn.functional as F

class SynchronizedGroupNorm(nn.Module):
    """
    GroupNorm whose statistics are computed jointly over
    both the batch (inter-view) and spatial dims—exactly
    what CubeDiff Sec 4.2 needs to keep all 6 faces color-consistent.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups   = num_groups
        self.num_channels = num_channels
        self.eps          = eps
        self.affine       = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        If x.ndim==4:  sync-GroupNorm over batch+spatial dims (CubeDiff Sec 4.2).
        If x.ndim==3:  fallback to regular per-sample GroupNorm on [B,C,L].
        """
        if x.ndim == 4:
            B, C, H, W = x.shape

            # fold batch into channels → [1, B*C, H, W]
            # x_fold = x.view(1, B * C, H, W)
            # use reshape to support non‑contiguous inputs
            x_fold = x.reshape(1, B * C, H, W)
            
            # sync-GroupNorm across (batch × channel) with G×B groups
            y_fold = F.group_norm(
                x_fold,
                num_groups=self.num_groups * B,
                # weight=w,
                # bias=b,
                weight=self.weight.repeat(B) if self.affine else None,
                bias=self.bias.repeat(B) if self.affine else None,
                eps=self.eps,
            )

            # unfold back → [B, C, H, W]
            return y_fold.reshape(C, B, H, W).permute(1, 0, 2, 3)

        elif x.ndim == 3:
            # [B, C, L] → classic per-sample GroupNorm
            return F.group_norm(
                x,
                num_groups=self.num_groups,
                # weight=w,
                # bias=b,
                weight=self.weight if self.affine else None,
                bias=self.bias if self.affine else None,
                eps=self.eps,
            )

        else:
            raise ValueError(f"SynchronizedGroupNorm expected 3D or 4D input, got {x.ndim}D")


# SGN (synchromized grouped normalization) is a special case of BatchNorm that normalizes across groups of channels.
def replace_group_norms(root_module: nn.Module, in_place: bool) -> int:
    """
    Recursively replace every nn.GroupNorm in root_module
    with SynchronizedGroupNorm, returning the number replaced.
    """
    replaced = 0
    for name, child in list(root_module.named_children()):
        if isinstance(child, nn.GroupNorm):
            # Get the device and dtype from the original module's parameters
            if child.affine:
                original_device = child.weight.device
                original_dtype = child.weight.dtype
            else:
                # Fallback if the layer has no parameters
                original_device = next(root_module.parameters()).device
                original_dtype = next(root_module.parameters()).dtype

            syncgn = SynchronizedGroupNorm(
                num_groups=child.num_groups,
                num_channels=child.num_channels,
                eps=child.eps,
                affine=child.affine,
            )
            # copy learned affine params
            # copies the values from the old GPU tensor to the new CPU tensor, but it does not change the device of syncgn.weight
            if child.affine:
                syncgn.weight.data.copy_(child.weight.data)
                syncgn.bias.data.copy_(child.bias.data)

            # Move the entire new module to the correct device and dtype
            # When the VAE's forward pass is called, the input tensor x is on the GPU, 
            # but it eventually reaches your SynchronizedGroupNorm layer, if the SGN self.weight parameter is still on the CPU. This will triggers an error.
            syncgn = syncgn.to(device=original_device, dtype=original_dtype)

            setattr(root_module, name, syncgn)
            replaced += 1
        else:
            replaced += replace_group_norms(child, True)
    return replaced


