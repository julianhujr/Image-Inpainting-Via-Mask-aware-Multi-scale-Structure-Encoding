"""
Quick training script with config file
Usage: python quick_train.py --config default
"""

import argparse
from config import get_config
from train import main as train_main


class Args:
    """Convert config to args object"""
    def __init__(self, config):
        # Data
        self.train_root = config.train_root
        self.val_root = config.val_root
        self.train_subset = config.train_subset
        self.val_subset = config.val_subset
        self.image_size = config.image_size
        
        # Model
        self.base_channels = config.base_channels
        
        # ✨ 新增: MSSE 相关参数
        self.use_structure_encoder = getattr(config, 'use_structure_encoder', True)
        
        # Training
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.num_workers = config.num_workers
        
        # Loss
        self.lambda_coarse = config.lambda_coarse
        self.lambda_refine = config.lambda_refine
        self.lambda_edge = config.lambda_edge
        self.lambda_color = config.lambda_color
        self.boundary_width = config.boundary_width
        self.lambda_corner = getattr(config, 'lambda_corner', 0.3)
        self.corner_threshold = getattr(config, 'corner_threshold', 0.2)
        
        # Checkpointing
        self.save_dir = config.save_dir
        self.resume = config.resume
        self.val_interval = config.val_interval
        self.save_interval = config.save_interval
        self.log_interval = config.log_interval
        
        # Metrics
        self.use_lpips = config.use_lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick Training with Config')
    parser.add_argument(
        '--config', 
        type=str, 
        default='default',
        choices=['default', 'high_quality', 'fast', 'large', 'small_gpu', 'debug'],
        help='Configuration preset'
    )
    parser.add_argument(
        '--train_root',
        type=str,
        default=None,
        help='Override train data path'
    )
    parser.add_argument(
        '--val_root',
        type=str,
        default=None,
        help='Override val data path'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Override save directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--no_structure_encoder',
        action='store_true',
        help='Disable MSSE module'
    )
    
    cmd_args = parser.parse_args()
    
    # Load config
    config = get_config(cmd_args.config)
    print(config)
    
    # Override with command line args
    if cmd_args.train_root:
        config.train_root = cmd_args.train_root
    if cmd_args.val_root:
        config.val_root = cmd_args.val_root
    if cmd_args.save_dir:
        config.save_dir = cmd_args.save_dir
    if cmd_args.resume:
        config.resume = cmd_args.resume
    if cmd_args.no_structure_encoder:
        config.use_structure_encoder = False
    
    # Convert to args object
    args = Args(config)
    
    # Start training
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Config preset:    {cmd_args.config}")
    print(f"Train data:       {args.train_root}")
    print(f"Val data:         {args.val_root}")
    print(f"Save directory:   {args.save_dir}")
    print(f"Model size:       {args.base_channels} channels ")
    print(f"MSSE enabled:     {args.use_structure_encoder}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print("="*70 + "\n")
    
    train_main(args)