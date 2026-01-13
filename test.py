import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np

from models import InpaintingNetwork
from dataset import InpaintingDataset
from metrics import MetricsCalculator, lpips_score
from utils import load_checkpoint, save_comparison_images, visualize_attention_maps


@torch.no_grad()
def test_model(model, test_loader, device, save_dir, lpips_model=None, save_attention=False):
    """Test the model and compute metrics"""
    model.eval()
    
    metrics_calculator = MetricsCalculator(device)
    all_metrics = []
    
    # For visualization
    vis_samples = []
    max_vis_samples = 16
    
    pbar = tqdm(test_loader, desc='Testing')
    
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(image, mask)
        
        # Compute metrics
        pred = outputs['output']
        metrics = metrics_calculator.calculate(pred, image, mask)
        
        # Add LPIPS if available
        if lpips_model is not None:
            try:
                metrics['lpips_mask'] = lpips_score(pred, image, mask, lpips_model)
            except:
                pass
        
        all_metrics.append(metrics)
        
        # Collect samples for visualization
        if len(vis_samples) < max_vis_samples:
            vis_samples.append({
                'image': image.cpu(),
                'mask': mask.cpu(),
                'masked_input': (image * (1 - mask)).cpu(),
                'coarse': outputs['coarse'].cpu(),
                'refined': outputs['refined'].cpu(),
                'output': outputs['output'].cpu(),
            })
            
            # Save attention maps for first few samples
            if save_attention and len(vis_samples) <= 4:
                attn_path = save_dir / f'attention_sample_{len(vis_samples)}.png'
                visualize_attention_maps(outputs['attentions'], attn_path)
        
        # Update progress bar
        pbar.set_postfix({
            'psnr_mask': f'{metrics["psnr_mask"]:.2f}',
            'ssim_mask': f'{metrics["ssim_mask"]:.4f}'
        })
    
    # Average metrics
    avg_metrics = metrics_calculator.average_metrics(all_metrics)
    
    # Print results
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    for key, value in avg_metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*50)
    
    # Save visualizations
    if vis_samples:
        save_path = save_dir / 'test_results.png'
        save_comparison_images(vis_samples, save_path, max_samples=16)
        print(f"\nVisualization saved to {save_path}")
    
    return avg_metrics


def inference_single_image(model, image_path, mask_path, device, save_path):
    """Run inference on a single image"""
    from PIL import Image
    import torchvision.transforms as T
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img = img_transform(img).unsqueeze(0).to(device)
    
    # Load or generate mask
    if mask_path:
        mask = Image.open(mask_path).convert('L')
        mask = T.Resize((256, 256))(mask)
        mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float().unsqueeze(0).to(device)
    else:
        # Generate random mask
        from dataset import random_free_form_mask
        mask = random_free_form_mask(256, 256)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(img, mask)
    
    # Save result
    sample = {
        'image': img.cpu(),
        'mask': mask.cpu(),
        'masked_input': (img * (1 - mask)).cpu(),
        'coarse': outputs['coarse'].cpu(),
        'refined': outputs['refined'].cpu(),
        'output': outputs['output'].cpu(),
    }
    
    save_comparison_images([sample], save_path, max_samples=1)
    print(f"Result saved to {save_path}")
    
    # Also save individual images
    from utils import denormalize
    import matplotlib.pyplot as plt
    
    output_dir = Path(save_path).parent
    
    # Save final output
    final_img = denormalize(outputs['output'][0]).permute(1, 2, 0).cpu().numpy()
    plt.imsave(output_dir / 'output.png', np.clip(final_img, 0, 1))
    
    # Save masked input
    masked_img = denormalize(img * (1 - mask))[0].permute(1, 2, 0).cpu().numpy()
    plt.imsave(output_dir / 'masked_input.png', np.clip(masked_img, 0, 1))
    
    print(f"Individual images saved to {output_dir}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = InpaintingNetwork(base_channels=args.base_channels).to(device)
    
    if args.checkpoint:
        epoch, best_metric = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {epoch}, best metric: {best_metric:.2f}")
    else:
        print("Warning: No checkpoint loaded, using random weights!")
    
    # LPIPS model (optional)
    lpips_model = None
    if args.use_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(device)
            lpips_model.eval()
            print("LPIPS metric enabled")
        except ImportError:
            print("Warning: lpips not installed, skipping LPIPS metric")
    
    # Single image inference or batch testing
    if args.image_path:
        # Single image inference
        inference_single_image(
            model, 
            args.image_path, 
            args.mask_path, 
            device, 
            save_dir / 'inference_result.png'
        )
    else:
        # Batch testing
        test_dataset = InpaintingDataset(
            args.test_root,
            image_size=args.image_size,
            mask_type='mixed',
            subset_size=args.test_subset
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        test_model(
            model, 
            test_loader, 
            device, 
            save_dir, 
            lpips_model, 
            save_attention=args.save_attention
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Inpainting Network')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels')
    
    # Single image inference
    parser.add_argument('--image_path', type=str, default=None, help='Path to single image for inference')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask (optional, will generate if not provided)')
    
    # Batch testing
    parser.add_argument('--test_root', type=str, default=None, help='Path to test images')
    parser.add_argument('--test_subset', type=int, default=None, help='Number of test images')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Options
    parser.add_argument('--save_dir', type=str, default='./test_results', help='Save directory')
    parser.add_argument('--use_lpips', action='store_true', help='Use LPIPS metric')
    parser.add_argument('--save_attention', action='store_true', help='Save attention maps')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.test_root:
        parser.error("Either --image_path or --test_root must be provided")
    
    main(args)