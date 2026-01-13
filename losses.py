import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SobelEdgeDetector(nn.Module):
    """Sobel edge detection"""
    def __init__(self):
        super(SobelEdgeDetector, self).__init__()
        
        # Create Sobel kernels as tensors
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        
        # Register as buffers (NOT parameters)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x):
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        sobel_x = self.sobel_x.to(gray.device)
        sobel_y = self.sobel_y.to(gray.device)

        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge


class LaplacianEdgeDetector(nn.Module):
    """Laplacian edge detection (for detecting corners/high-curvature regions)"""
    def __init__(self):
        super(LaplacianEdgeDetector, self).__init__()
        
        # Laplacian kernel
        laplacian = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)
    
    def forward(self, x):
        if x.size(1) == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        laplacian = self.laplacian.to(gray.device)
        edge = F.conv2d(gray, laplacian, padding=1)
        edge = torch.abs(edge)
        
        return edge


class StructureAwareAttentionBias(nn.Module):
    """Generate structure-aware bias for attention mechanism"""
    def __init__(self, edge_threshold=0.1, lambda_struct=0.15):
        super().__init__()
        self.edge_detector = SobelEdgeDetector()
        self.edge_threshold = edge_threshold
        self.lambda_struct = lambda_struct
    
    def forward(self, image, mask):
        """
        image: [B, 3, H, W]
        mask: [B, 1, H, W], 1=missing
        Returns: attention bias [B, 1, H, W]
        """
        # Detect edges
        edges = self.edge_detector(image)  # [B, 1, H, W]
        
        # Normalize edges
        edges = edges / (edges.max() + 1e-8)
        
        # Create bias: encourage attending to high-edge regions
        # But only within a reasonable distance from mask boundary
        bias = torch.zeros_like(mask)
        
        # High-edge regions get positive bias
        high_edge_regions = (edges > self.edge_threshold).float()
        bias = high_edge_regions * self.lambda_struct
        
        return bias


class InpaintingLoss(nn.Module):
    """Complete loss function for inpainting with detail constraints"""
    def __init__(self, lambda_coarse=1.0, lambda_refine=1.0, 
                 lambda_edge=0.5, lambda_color=0.5, boundary_width=5,
                 lambda_corner=0.3, corner_threshold=0.2):
        super(InpaintingLoss, self).__init__()
        
        self.lambda_coarse = lambda_coarse
        self.lambda_refine = lambda_refine
        self.lambda_edge = lambda_edge
        self.lambda_color = lambda_color
        self.boundary_width = boundary_width
        self.lambda_corner = lambda_corner  # New: corner loss weight
        self.corner_threshold = corner_threshold  # New: threshold for corner detection
        
        self.l1_loss = nn.L1Loss()
        self.edge_detector = SobelEdgeDetector()
        self.laplacian_detector = LaplacianEdgeDetector()
        
    def compute_boundary_mask(self, mask, width=5):
        """
        Compute boundary region of mask
        mask: [B, 1, H, W], 1=missing
        Returns: boundary mask [B, 1, H, W]
        """
        kernel_size = width * 2 + 1
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=width)
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=width)
        boundary = dilated - eroded
        return boundary
    
    def compute_corner_weight_map(self, gt_image, tau=0.2, beta=2.0):
        """
        Compute corner-aware weight map
        高梯度区域（眼角、嘴角、边界）的权重更高
        
        gt_image: [B, 3, H, W]
        tau: gradient threshold for corner detection
        beta: weight amplification factor
        Returns: weight map [B, 1, H, W]
        """
        # Compute Laplacian (detects corners and high-curvature regions)
        laplacian = self.laplacian_detector(gt_image)  # [B, 1, H, W]
        
        # Normalize
        laplacian = laplacian / (laplacian.max() + 1e-8)
        
        # Also compute Sobel gradient magnitude
        edge = self.edge_detector(gt_image)  # [B, 1, H, W]
        edge = edge / (edge.max() + 1e-8)
        
        # Combine: prioritize high-curvature, high-gradient regions
        corner_indicator = torch.clamp(laplacian + edge, 0, 2)  # [B, 1, H, W]
        
        # Create weight: base 1.0 + amplification in corner regions
        weight = 1.0 + beta * (corner_indicator > tau).float()
        
        return weight
    
    def forward(self, outputs, gt_image, mask, epoch=0, total_epochs=100):
        """
        outputs: dict with 'coarse', 'refined', 'output'
        gt_image: ground truth [B, 3, H, W]
        mask: binary mask [B, 1, H, W], 1=missing
        epoch: current epoch (for progressive weighting)
        total_epochs: total training epochs
        """
        coarse = outputs['coarse']
        refined = outputs['refined']
        output = outputs['output']
        
        # 1. Coarse L1 loss (only in mask region)
        loss_coarse = torch.mean(torch.abs(mask * (coarse - gt_image)))
        
        # 2. Refinement L1 loss (only in mask region)
        loss_refine = torch.mean(torch.abs(mask * (refined - gt_image)))
        
        # 3. Edge consistency loss (standard)
        edge_pred = self.edge_detector(output)
        edge_gt = self.edge_detector(gt_image)

        # Apply edge loss ONLY in masked + boundary region
        boundary_mask = self.compute_boundary_mask(mask, self.boundary_width)
        struct_mask = torch.clamp(mask + boundary_mask, 0, 1)

        loss_edge = torch.sum(
            torch.abs(edge_pred - edge_gt) * struct_mask
        ) / (struct_mask.sum() + 1e-8)

        # Corner-aware weighting (masked)
        corner_weight = self.compute_corner_weight_map(
            gt_image, tau=self.corner_threshold, beta=2.0
        )

        loss_corner = torch.sum(
            torch.abs(edge_pred - edge_gt) * corner_weight * struct_mask
        ) / (struct_mask.sum() + 1e-8)

        
        # 4. Color consistency loss (boundary region)
        boundary_mask = self.compute_boundary_mask(mask, self.boundary_width)
        
        # Mean and std in boundary region
        if boundary_mask.sum() > 0:
            pred_boundary = output * boundary_mask
            gt_boundary = gt_image * boundary_mask
            
            # Compute mean
            mask_sum = boundary_mask.sum(dim=[2, 3], keepdim=True) + 1e-8

            pred_mean = pred_boundary.sum(dim=[2, 3], keepdim=True) / mask_sum
            gt_mean = gt_boundary.sum(dim=[2, 3], keepdim=True) / mask_sum

            # Compute std
            pred_std = torch.sqrt(((pred_boundary - pred_mean)**2 * boundary_mask).sum(dim=[2, 3], keepdim=True) / 
                                (boundary_mask.sum(dim=[2, 3], keepdim=True) + 1e-8) + 1e-8)
            gt_std = torch.sqrt(((gt_boundary - gt_mean)**2 * boundary_mask).sum(dim=[2, 3], keepdim=True) / 
                              (boundary_mask.sum(dim=[2, 3], keepdim=True) + 1e-8) + 1e-8)
            
            loss_color = torch.mean(torch.abs(pred_mean - gt_mean)) + torch.mean(torch.abs(pred_std - gt_std))
        else:
            loss_color = torch.tensor(0.0, device=output.device)
        
        # Progressive weighting schedule
        progress = epoch / max(total_epochs, 1)
        
        w_coarse = self.lambda_coarse * (1.0 - 0.5 * progress)
        w_refine = self.lambda_refine
        w_edge = self.lambda_edge * (0.5 + 0.5 * progress)
        w_color = self.lambda_color * (0.5 + 0.5 * progress)
        w_corner = self.lambda_corner * (0.3 + 0.7 * progress)  # Grow from 0.3 to 1.0
        
        # Total loss
        total_loss = (w_coarse * loss_coarse + 
                     w_refine * loss_refine + 
                     w_edge * loss_edge +
                     w_corner * loss_corner +
                     w_color * loss_color)
        
        return {
            'total': total_loss,
            'coarse': loss_coarse,
            'refine': loss_refine,
            'edge': loss_edge,
            'corner': loss_corner,  # New metric
            'color': loss_color
        }


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss (optional, for better quality)"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, gt):
        """Compute perceptual loss"""
        # Normalize to ImageNet range
        pred = (pred + 1) / 2  # [-1, 1] -> [0, 1]
        gt = (gt + 1) / 2
        
        # Extract features
        pred_feats = []
        gt_feats = []
        
        for layer in [self.slice1, self.slice2, self.slice3]:
            pred = layer(pred)
            gt = layer(gt)
            pred_feats.append(pred)
            gt_feats.append(gt)
        
        # Compute L1 loss on features
        loss = 0
        for pf, gf in zip(pred_feats, gt_feats):
            loss += F.l1_loss(pf, gf)
        
        return loss / len(pred_feats)


if __name__ == "__main__":
    # Test losses
    print("Testing improved losses on CUDA...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    criterion = InpaintingLoss()
    criterion = criterion.to(device)
    
    # Dummy data
    batch_size = 2
    img = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256)).float().to(device)
    
    outputs = {
        'coarse': torch.randn(batch_size, 3, 256, 256).to(device),
        'refined': torch.randn(batch_size, 3, 256, 256).to(device),
        'output': torch.randn(batch_size, 3, 256, 256).to(device)
    }
    
    print("\nTesting loss computation...")
    try:
        losses = criterion(outputs, img, mask, epoch=0)
        print("✓ Success! Losses:")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()