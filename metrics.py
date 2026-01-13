import torch
import torch.nn.functional as F
import numpy as np
from math import exp


def psnr(pred, gt, mask=None):
    """
    Calculate PSNR
    pred, gt: [B, C, H, W], range [-1, 1]
    mask: [B, 1, H, W], 1=region to evaluate, None=full image
    """
    # Convert to [0, 1]
    pred = (pred + 1) / 2
    gt = (gt + 1) / 2
    
    if mask is not None:
        # Only compute in masked region
        mse = ((pred - gt) ** 2 * mask).sum() / (mask.sum() * pred.size(1) + 1e-8)
    else:
        mse = F.mse_loss(pred, gt)
    
    if mse < 1e-10:
        return 100.0
    
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    """Create Gaussian window"""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(pred, gt, mask=None, window_size=11, size_average=True):
    """
    Calculate SSIM
    pred, gt: [B, C, H, W], range [-1, 1]
    mask: [B, 1, H, W], 1=region to evaluate
    """
    # Convert to [0, 1]
    pred = (pred + 1) / 2
    gt = (gt + 1) / 2
    
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)
    
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if mask is not None:
        # Only compute in masked region
        mask_resized = F.interpolate(mask, size=ssim_map.shape[2:], mode='nearest')
        ssim_value = (ssim_map * mask_resized).sum() / (mask_resized.sum() + 1e-8)
    else:
        if size_average:
            ssim_value = ssim_map.mean()
        else:
            ssim_value = ssim_map.mean(1).mean(1).mean(1)
    
    return ssim_value


def boundary_psnr(pred, gt, mask, width=5):
    """
    Calculate PSNR in boundary region
    pred, gt: [B, C, H, W]
    mask: [B, 1, H, W], 1=missing region
    width: boundary width in pixels
    """
    # Compute boundary mask
    kernel_size = width * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=width)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=width)
    boundary = dilated - eroded
    
    return psnr(pred, gt, boundary)


class MetricsCalculator:
    """Calculate multiple metrics"""
    def __init__(self, device='cuda'):
        self.device = device
        
    def calculate(self, pred, gt, mask):
        """
        Calculate all metrics
        pred, gt: [B, C, H, W], range [-1, 1]
        mask: [B, 1, H, W], 1=missing
        Returns: dict of metrics
        """
        with torch.no_grad():
            metrics = {}
            
            # Full image metrics
            metrics['psnr_full'] = psnr(pred, gt, mask=None).item()
            metrics['ssim_full'] = ssim(pred, gt, mask=None).item()
            
            # Mask region metrics
            metrics['psnr_mask'] = psnr(pred, gt, mask=mask).item()
            metrics['ssim_mask'] = ssim(pred, gt, mask=mask).item()
            
            # Boundary metrics
            metrics['psnr_boundary'] = boundary_psnr(pred, gt, mask, width=5).item()
            
        return metrics
    
    def average_metrics(self, metrics_list):
        """Average a list of metric dicts"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics


def lpips_score(pred, gt, mask=None, lpips_model=None):
    """
    Masked LPIPS for inpainting (CORRECT VERSION)

    pred, gt: [B, 3, H, W], range [-1, 1]
    mask: [B, 1, H, W], 1 = missing region
    """
    try:
        import lpips

        if lpips_model is None:
            lpips_model = lpips.LPIPS(net='alex').to(pred.device)
            lpips_model.eval()

        with torch.no_grad():
            if mask is not None:
                # Expand mask to 3 channels
                mask_3ch = mask.repeat(1, 3, 1, 1)

                # Mask BEFORE LPIPS
                pred = pred * mask_3ch
                gt = gt * mask_3ch

            score = lpips_model(pred, gt)

            return score.mean().item()

    except ImportError:
        print("Warning: lpips package not installed. Install with: pip install lpips")
        return 0.0



if __name__ == "__main__":
    # Test metrics
    calculator = MetricsCalculator()
    
    # Dummy data
    pred = torch.randn(2, 3, 256, 256).cuda()
    gt = torch.randn(2, 3, 256, 256).cuda()
    mask = torch.randint(0, 2, (2, 1, 256, 256)).float().cuda()
    
    metrics = calculator.calculate(pred, gt, mask)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")