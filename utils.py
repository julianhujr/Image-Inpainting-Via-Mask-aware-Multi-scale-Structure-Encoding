import torch
import torchvision.utils as vutils
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_metric):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0)
    
    return epoch, best_metric


def denormalize(tensor):
    """Denormalize from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def save_comparison_images(samples, save_path, max_samples=8):
    """
    Save comparison grid of images
    samples: list of dicts containing 'image', 'mask', 'masked_input', 'coarse', 'refined', 'output'
    """
    samples = samples[:max_samples]
    n_samples = len(samples)
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    titles = ['Ground Truth', 'Mask', 'Masked Input', 'Coarse', 'Refined', 'Final Output']
    
    for i, sample in enumerate(samples):
        # Ground truth
        img = denormalize(sample['image'][0]).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(titles[0])
        
        # Mask (show in red)
        mask = sample['mask'][0, 0].numpy()
        mask_vis = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        axes[i, 1].imshow(mask_vis)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(titles[1])
        
        # Masked input
        masked = denormalize(sample['masked_input'][0]).permute(1, 2, 0).numpy()
        axes[i, 2].imshow(np.clip(masked, 0, 1))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title(titles[2])
        
        # Coarse
        coarse = denormalize(sample['coarse'][0]).permute(1, 2, 0).numpy()
        axes[i, 3].imshow(np.clip(coarse, 0, 1))
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title(titles[3])
        
        # Refined
        refined = denormalize(sample['refined'][0]).permute(1, 2, 0).numpy()
        axes[i, 4].imshow(np.clip(refined, 0, 1))
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title(titles[4])
        
        # Final output
        output = denormalize(sample['output'][0]).permute(1, 2, 0).numpy()
        axes[i, 5].imshow(np.clip(output, 0, 1))
        axes[i, 5].axis('off')
        if i == 0:
            axes[i, 5].set_title(titles[5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to {save_path}")


def visualize_attention_maps(attentions, save_path):
    """
    Visualize attention maps from the refinement stages
    attentions: list of [attn1, attn2, attn3], each [B, num_heads, HW, HW]
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stage_names = ['Stage 1: Structure', 'Stage 2: Mid-level', 'Stage 3: Texture']
    
    for i, (attn, name) in enumerate(zip(attentions, stage_names)):
        # Take first sample, first head, and average over query positions
        attn_map = attn[0, 0].cpu().numpy()  # [HW, HW]
        
        # Reshape to spatial
        hw = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map.mean(axis=0).reshape(hw, hw)
        
        # Plot
        im = axes[i].imshow(attn_map, cmap='hot', interpolation='bilinear')
        axes[i].set_title(name)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_model_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Create dummy sample
    sample = {
        'image': torch.randn(1, 3, 256, 256),
        'mask': torch.randint(0, 2, (1, 1, 256, 256)).float(),
        'masked_input': torch.randn(1, 3, 256, 256),
        'coarse': torch.randn(1, 3, 256, 256),
        'refined': torch.randn(1, 3, 256, 256),
        'output': torch.randn(1, 3, 256, 256)
    }
    
    save_comparison_images([sample], './test_comparison.png')
    print("Test comparison saved!")