import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from models import InpaintingNetwork
from dataset import get_dataloaders
from losses import InpaintingLoss
from metrics import MetricsCalculator, lpips_score
from metric_tracker import MetricsTracker
from utils import save_checkpoint, load_checkpoint, save_comparison_images


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, writer, log_interval=50):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_components = {'coarse': 0, 'refine': 0, 'edge': 0, 'corner': 0, 'color': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(image, mask)
        
        # Compute loss
        losses = criterion(
            outputs, image, mask,
            epoch=total_epochs,  # force final weights
            total_epochs=total_epochs
        )
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coarse': f'{losses["coarse"].item():.4f}',
            'refine': f'{losses["refine"].item():.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
            writer.add_scalar('Train/Loss_Coarse', losses['coarse'].item(), global_step)
            writer.add_scalar('Train/Loss_Refine', losses['refine'].item(), global_step)
            writer.add_scalar('Train/Loss_Edge', losses['edge'].item(), global_step)
            writer.add_scalar('Train/Loss_Corner', losses['corner'].item(), global_step)
            writer.add_scalar('Train/Loss_Color', losses['color'].item(), global_step)
    
    # Average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, total_epochs, writer, save_dir, lpips_model=None):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    metrics_calculator = MetricsCalculator(device)
    all_metrics = []
    
    # For visualization
    vis_samples = []
    max_vis_samples = 8
    
    pbar = tqdm(val_loader, desc='Validation')
    
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(image, mask)
        
        # Compute loss
        losses = criterion(outputs, image, mask, epoch=epoch, total_epochs=total_epochs)
        total_loss += losses['total'].item()
        
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
        
        # Update progress bar
        pbar.set_postfix({
            'psnr_mask': f'{metrics["psnr_mask"]:.2f}',
            'ssim_mask': f'{metrics["ssim_mask"]:.4f}'
        })
    
    # Average metrics
    avg_metrics = metrics_calculator.average_metrics(all_metrics)
    avg_loss = total_loss / len(val_loader)
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    for key, value in avg_metrics.items():
        writer.add_scalar(f'Val/{key}', value, epoch)
    
    # Save visualization
    if vis_samples:
        save_path = save_dir / f'val_epoch_{epoch:04d}.png'
        save_comparison_images(vis_samples, save_path)
        
        # Log to tensorboard
        import torchvision
        grid = torchvision.utils.make_grid(
            torch.cat([
                vis_samples[0]['image'],
                vis_samples[0]['masked_input'],
                vis_samples[0]['coarse'],
                vis_samples[0]['output']
            ], dim=0),
            nrow=4,
            normalize=True,
            value_range=(-1, 1)
        )
        writer.add_image('Val/Comparison', grid, epoch)
    
    return avg_loss, avg_metrics


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    ckpt_dir = save_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=save_dir / 'logs')
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(save_dir=save_dir / 'metrics_plots', plot_interval=3)
    
    # ✨ 修改：传入use_structure_encoder参数
    model = InpaintingNetwork(
        base_channels=args.base_channels,
        use_structure_encoder=args.use_structure_encoder  # 新增参数
    ).to(device)
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    print(f"Structure Encoder: {'Enabled' if args.use_structure_encoder else 'Disabled'}")
    
    # Loss and optimizer
    criterion = InpaintingLoss(
        lambda_coarse=args.lambda_coarse,
        lambda_refine=args.lambda_refine,
        lambda_edge=args.lambda_edge,
        lambda_color=args.lambda_color,
        boundary_width=args.boundary_width,
        lambda_corner=getattr(args, 'lambda_corner', 0.3),
        corner_threshold=getattr(args, 'corner_threshold', 0.2)
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Data loaders
    train_loader, val_loader = get_dataloaders(
        train_root=args.train_root,
        val_root=args.val_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        val_subset=args.val_subset
    )
    
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
    
    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_loss_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, writer, args.log_interval
        )
        
        # Log training metrics
        metrics_tracker.log_train_metrics(epoch, {
            'total': train_loss,
            'coarse': train_loss_components['coarse'],
            'refine': train_loss_components['refine'],
            'edge': train_loss_components['edge'],
            'corner': train_loss_components['corner']
        })
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Coarse: {train_loss_components['coarse']:.4f}")
        print(f"    Refine: {train_loss_components['refine']:.4f}")
        print(f"    Edge: {train_loss_components['edge']:.4f}")
        print(f"    Corner: {train_loss_components['corner']:.4f}")
        print(f"    Color: {train_loss_components['color']:.4f}")
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device,
                epoch, args.epochs, writer, vis_dir, lpips_model
            )
            
            # Log validation metrics
            metrics_tracker.log_val_metrics(epoch, {
                'psnr_mask': val_metrics.get('psnr_mask', 0),
                'ssim_mask': val_metrics.get('ssim_mask', 0),
                'psnr_boundary': val_metrics.get('psnr_boundary', 0)
            })
            
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics:")
            for key, value in val_metrics.items():
                print(f"    {key}: {value:.4f}")
            
            # Save best model
            current_psnr = val_metrics.get('psnr_mask', 0)
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                save_checkpoint(
                    ckpt_dir / 'best_model.pth',
                    epoch, model, optimizer, scheduler, best_psnr
                )
                print(f"  ✓ Best model saved (PSNR: {best_psnr:.2f})")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                ckpt_dir / f'checkpoint_epoch_{epoch:04d}.pth',
                epoch, model, optimizer, scheduler, best_psnr
            )
        
        # Plot metrics
        metrics_tracker.plot_metrics(epoch)
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
    
    # Save final model
    save_checkpoint(
        ckpt_dir / 'final_model.pth',
        args.epochs - 1, model, optimizer, scheduler, best_psnr
    )
    
    # Print summary
    metrics_tracker.print_summary()
    
    print(f"\nTraining completed! Best PSNR: {best_psnr:.2f}")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Inpainting Network with MSSE')
    
    # Data
    parser.add_argument('--train_root', type=str, required=True, help='Path to training images')
    parser.add_argument('--val_root', type=str, required=True, help='Path to validation images')
    parser.add_argument('--train_subset', type=int, default=10000, help='Number of training images')
    parser.add_argument('--val_subset', type=int, default=1000, help='Number of validation images')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels')
    
    # Training
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Loss weights
    parser.add_argument('--lambda_coarse', type=float, default=1.0, help='Coarse loss weight')
    parser.add_argument('--lambda_refine', type=float, default=1.0, help='Refinement loss weight')
    parser.add_argument('--lambda_edge', type=float, default=0.5, help='Edge loss weight')
    parser.add_argument('--lambda_color', type=float, default=0.5, help='Color loss weight')
    parser.add_argument('--boundary_width', type=int, default=5, help='Boundary width')
    parser.add_argument('--lambda_corner', type=float, default=0.3, help='Corner loss weight')
    parser.add_argument('--corner_threshold', type=float, default=0.2, help='Corner threshold')
    
    # ✨ 新增：MSSE相关参数
    parser.add_argument('--use_structure_encoder', action='store_true', 
                       default=True, help='Enable MultiScaleStructureEncoder')
    parser.add_argument('--no_structure_encoder', action='store_false',
                       dest='use_structure_encoder', help='Disable MSSE')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./experiments/exp1', help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--log_interval', type=int, default=50, help='Log interval')
    
    # Metrics
    parser.add_argument('--use_lpips', action='store_true', help='Use LPIPS metric')
    
    args = parser.parse_args()
    
    main(args)