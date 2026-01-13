import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Correct Partial Convolution (Liu et al. 2018)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        self.mask_conv = nn.Conv2d(
            1, 1, kernel_size, stride, padding, bias=False
        )

        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask):
        """
        x: [B, C, H, W]
        mask: [B, 1, H, W], 1=valid, 0=missing
        """
        with torch.no_grad():
            # number of valid pixels in each window
            valid_count = self.mask_conv(mask)
            updated_mask = (valid_count > 0).float()

        # masked convolution
        x = x * mask
        out = self.conv(x)

        # correct normalization
        out = torch.where(
            updated_mask.bool(),
            out / (valid_count + 1e-8),
            torch.zeros_like(out)
        )

        out = self.bn(out)
        return out, updated_mask



class ContextEncoder(nn.Module):
    """Encoder with Partial Convolution for extracting context features"""
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        # Encoder blocks with partial convolution
        # Level 1: base_channels (shallow features)
        self.enc1 = nn.ModuleList([
            PartialConv2d(in_channels, base_channels, 7, 1, 3),
            nn.ReLU(inplace=True),
            PartialConv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])
        
        # Level 2: base_channels*2 (mid-level features)
        self.enc2 = nn.ModuleList([
            PartialConv2d(base_channels, base_channels*2, 5, 2, 2),
            nn.ReLU(inplace=True),
            PartialConv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            PartialConv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])
        
        # Level 3: base_channels*4 (deep features)
        self.enc3 = nn.ModuleList([
            PartialConv2d(base_channels*2, base_channels*4, 3, 2, 1),
            nn.ReLU(inplace=True),
            PartialConv2d(base_channels*4, base_channels*4, 3, 1, 1),
            nn.ReLU(inplace=True),
            PartialConv2d(base_channels*4, base_channels*4, 3, 1, 1),
            nn.ReLU(inplace=True),
            PartialConv2d(base_channels*4, base_channels*4, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])
        
    def forward(self, x, mask):
        """
        x: input image [B, 3, H, W]
        mask: binary mask [B, 1, H, W], 1=valid, 0=missing
        Returns: multi-scale context features
        """
        features = []
        
        # Level 1: shallow features (structure/edges)
        for layer in self.enc1:
            if isinstance(layer, PartialConv2d):
                x, mask = layer(x, mask)
            else:
                x = layer(x)
        f1 = x
        features.append(f1)
        
        # Level 2: mid-level features (local shape)
        for layer in self.enc2:
            if isinstance(layer, PartialConv2d):
                x, mask = layer(x, mask)
            else:
                x = layer(x)
        f2 = x
        features.append(f2)
        
        # Level 3: deep features (texture/color)
        for layer in self.enc3:
            if isinstance(layer, PartialConv2d):
                x, mask = layer(x, mask)
            else:
                x = layer(x)
        f3 = x
        features.append(f3)
        
        return features


class CoarseDecoder(nn.Module):
    """Simple decoder for coarse inpainting"""
    def __init__(self, in_channels=128, base_channels=32):
        super().__init__()
        
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels*4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.out(x)
        return x


class MaskQueryCrossAttention(nn.Module):
    """
    Cross-attention between mask query and context features (memory-efficient version)
    现在支持 structure_bias 注入
    """
    def __init__(self, query_dim, ctx_dim, num_heads=4, kernel_size=7, use_structure_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.kernel_size = kernel_size
        self.use_structure_bias = use_structure_bias
        
        self.q_proj = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, 1)
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(ctx_dim, query_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, 1)
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(ctx_dim, query_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, 1)
        )
        
        # Local convolution to approximate attention
        padding = kernel_size // 2
        self.local_attn = nn.Sequential(
            nn.Conv2d(query_dim * 2, query_dim * 2, kernel_size, padding=padding, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim * 2, query_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, 1)
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(query_dim, query_dim, 1)
        )
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, query, context, structure_bias=None):
        """
        query: mask feature [B, C, H, W]
        context: context feature [B, C', H', W']
        structure_bias: 结构置信度图 [B, 1, H, W]，用于attention偏置
        """
        B, C, H, W = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [B, C, H, W]
        K = self.k_proj(context)  # [B, C, H', W']
        V = self.v_proj(context)  # [B, C, H', W']
        
        # Resize K, V to match Q spatial size if needed
        if K.shape[2:] != Q.shape[2:]:
            K = F.interpolate(K, size=(H, W), mode='bilinear', align_corners=False)
            V = F.interpolate(V, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate Q and V, use convolution to aggregate
        QV = torch.cat([Q, V], dim=1)  # [B, 2C, H, W]
        out = self.local_attn(QV)  # [B, C, H, W]
        
        # ✨ 关键改进：注入structure_bias作为attention偏置
        # 方式：modulation - out = out + α * structure_bias * Q
        if structure_bias is not None and self.use_structure_bias:
            out = out + 0.1 * structure_bias * Q
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual + LayerNorm
        out = out + query
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Create dummy attention map for compatibility
        attn = torch.zeros(B, self.num_heads, 1, 1, device=query.device)
        
        return out, attn


class ProgressiveRefinementModule(nn.Module):
    """
    Progressive multi-scale refinement with cross-attention
    现在集成 MultiScaleStructureEncoder (MSSE)
    """
    def __init__(self, base_channels=32, use_structure_encoder=True):
        super().__init__()
        
        self.use_structure_encoder = use_structure_encoder
        
        # Feature extraction from coarse result
        self.mask_feat_extract = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # ✨ 新增：MultiScaleStructureEncoder
        if use_structure_encoder:
            # 导入MSSE类（在msse.py中定义）
            from msse import MultiScaleStructureEncoder
            self.structure_encoder = MultiScaleStructureEncoder(
                in_channels=base_channels,
                patch_sizes=(3, 7, 15)
            )
        
        # Three stages of refinement with structure awareness
        # Stage 1: structure (attend to f1_ctx, 32 channels)
        self.attn1 = MaskQueryCrossAttention(
            base_channels, base_channels, num_heads=4, kernel_size=7, 
            use_structure_bias=use_structure_encoder
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: mid-level (attend to f2_ctx, 64 channels)
        self.attn2 = MaskQueryCrossAttention(
            base_channels, base_channels*2, num_heads=4, kernel_size=7,
            use_structure_bias=use_structure_encoder
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: texture (attend to f3_ctx, 128 channels)
        self.attn3 = MaskQueryCrossAttention(
            base_channels, base_channels*4, num_heads=4, kernel_size=7,
            use_structure_bias=use_structure_encoder
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, coarse_img, mask, ctx_features):
        """
        coarse_img: coarse inpainting result [B, 3, H, W]
        mask: binary mask [B, 1, H, W], 1=missing
        ctx_features: [f1_ctx, f2_ctx, f3_ctx]
        """
        # Extract mask feature from coarse result
        f0_mask = self.mask_feat_extract(coarse_img) * mask  # Only in mask region
        
        # ✨ 关键改进：从浅层特征生成结构置信度图
        structure_bias = None
        if self.use_structure_encoder:
            structure_bias = self.structure_encoder(ctx_features[0], mask)
            # structure_bias: [B, 1, H, W]，值在[0,1]之间
        
        # Stage 1: structure refinement
        # 使用同一个structure_bias在所有stages中
        f1_mask, attn1 = self.attn1(f0_mask, ctx_features[0], structure_bias)
        f1_mask = f0_mask + self.refine1(f1_mask)
        
        # Stage 2: mid-level refinement
        f2_mask, attn2 = self.attn2(f1_mask, ctx_features[1], structure_bias)
        f2_mask = f1_mask + self.refine2(f2_mask)
        
        # Stage 3: texture refinement
        f3_mask, attn3 = self.attn3(f2_mask, ctx_features[2], structure_bias)
        f3_mask = f2_mask + self.refine3(f3_mask)
        
        return f3_mask, [attn1, attn2, attn3]


class RefinementDecoder(nn.Module):
    """Final decoder combining refined features with coarse result"""
    def __init__(self, feat_channels=32):
        super().__init__()
        
        self.combine = nn.Sequential(
            nn.Conv2d(feat_channels + 3, feat_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels*2, feat_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, refined_feat, coarse_img):
        """
        refined_feat: refined mask feature [B, C, H, W]
        coarse_img: coarse result [B, 3, H, W]
        """
        x = torch.cat([refined_feat, coarse_img], dim=1)
        delta = self.combine(x)
        return delta


class InpaintingNetwork(nn.Module):
    """
    Complete inpainting network with coarse-to-fine refinement
    现在集成 MultiScaleStructureEncoder (MSSE)
    """
    def __init__(self, base_channels=32, use_structure_encoder=True):
        super().__init__()
        
        self.encoder = ContextEncoder(in_channels=3, base_channels=base_channels)
        self.coarse_decoder = CoarseDecoder(in_channels=base_channels*4, base_channels=base_channels)
        self.refinement = ProgressiveRefinementModule(
            base_channels=base_channels,
            use_structure_encoder=use_structure_encoder
        )
        self.refine_decoder = RefinementDecoder(feat_channels=base_channels)
        
    def forward(self, image, mask):
        """
        image: [B, 3, H, W]
        mask:  [B, 1, H, W], 1=missing
        """
        B, _, H, W = image.shape
        assert H % 4 == 0 and W % 4 == 0, \
            "Input size must be divisible by 4 for current encoder-decoder design"

        valid_mask = 1 - mask
        masked_img = image * valid_mask

        # Encoder
        ctx_features = self.encoder(masked_img, valid_mask)

        # Sanity check encoder scales
        f3 = ctx_features[2]
        assert f3.shape[2] == H // 4 and f3.shape[3] == W // 4, \
            "Encoder output spatial size mismatch"

        # Coarse inpainting
        coarse_result = self.coarse_decoder(f3)

        # Refinement
        refined_feat, attentions = self.refinement(
            coarse_result, mask, ctx_features
        )

        delta = self.refine_decoder(refined_feat, coarse_result)
        final_result = coarse_result + delta

        # Final composition (DESIGN INVARIANT)
        output = image * valid_mask + final_result * mask

        return {
            'coarse': coarse_result,
            'refined': final_result,
            'output': output,
            'attentions': attentions
        }

    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model with MSSE
    print("Testing InpaintingNetwork with MSSE...")
    model = InpaintingNetwork(base_channels=32, use_structure_encoder=True)
    print(f"Total parameters: {model.count_parameters() / 1e6:.2f}M")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    img = torch.randn(2, 3, 256, 256).to(device)
    mask = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    
    print(f"Input image shape: {img.shape}")
    print(f"Input mask shape: {mask.shape}")
    
    try:
        with torch.no_grad():
            output = model(img, mask)
        
        print("✓ Forward pass successful!")
        print(f"  Coarse shape: {output['coarse'].shape}")
        print(f"  Refined shape: {output['refined'].shape}")
        print(f"  Output shape: {output['output'].shape}")
        print(f"  Num attentions: {len(output['attentions'])}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()