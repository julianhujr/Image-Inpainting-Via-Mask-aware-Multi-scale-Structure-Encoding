import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleStructureEncoder(nn.Module):
    """
    Memory-safe MSSE
    输出:
        structure_map: [B, 1, H, W] ∈ [0,1]
    """

    def __init__(self, in_channels, patch_sizes=(3, 7, 15), hidden_ratio=0.25):
        super().__init__()

        self.patch_sizes = patch_sizes
        hidden_channels = int(in_channels * hidden_ratio)

        self.branches = nn.ModuleList()
        for p in patch_sizes:
            self.branches.append(
                nn.Sequential(
                    # patch-level descriptor
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=p, padding=p // 2, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),

                    # reduce to structure confidence
                    nn.Conv2d(hidden_channels, 1, kernel_size=1)
                )
            )

        # learnable scale fusion
        self.scale_weights = nn.Parameter(torch.ones(len(patch_sizes)))

    @staticmethod
    def compute_boundary_mask(mask, min_width=3, ratio=0.02):
        """
        自适应 boundary
        mask: [B,1,H,W], 1=missing
        """
        B, _, H, W = mask.shape
        width = max(min_width, int(ratio * min(H, W)))

        kernel_size = width * 2 + 1
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=width)
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=width)

        return (dilated - eroded).clamp(0, 1)
    
    def forward(self, feat, mask):
        """
        feat: [B, C, H, W]
        mask: [B, 1, H, W], 1=missing
        """
        boundary = self.compute_boundary_mask(mask)

        scale_weights = F.softmax(self.scale_weights, dim=0)

        structure_map = 0.0
        for w, branch in zip(scale_weights, self.branches):
            score = branch(feat)  # [B,1,H,W]
            structure_map = structure_map + w * score

        structure_map = torch.sigmoid(structure_map)

        # ✅ CRITICAL FIX: restrict to boundary AND missing region only
        structure_map = structure_map * boundary * mask

        return structure_map




class StructureAwareAttentionGate(nn.Module):
    def __init__(self, fusion_mode='modulation'):
        super().__init__()
        self.fusion_mode = fusion_mode

    def forward(self, attn_output, structure_bias, query_feat=None):
        if structure_bias is None:
            return attn_output

        if self.fusion_mode == 'modulation':
            base = query_feat if query_feat is not None else attn_output
            return attn_output + 0.1 * structure_bias * base

        elif self.fusion_mode == 'gating':
            return attn_output * (1.0 + 0.2 * structure_bias)

        elif self.fusion_mode == 'hybrid':
            base = query_feat if query_feat is not None else attn_output
            mod = attn_output + 0.1 * structure_bias * base
            return mod * (1.0 + 0.1 * structure_bias)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
