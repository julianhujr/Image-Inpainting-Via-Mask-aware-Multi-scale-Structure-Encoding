"""
Configuration file for easy training setup
Updated with MSSE parameters
"""

class Config:
    """Training configuration"""
    
    # ==================== Data ====================
    train_root = './data/train'
    val_root = './data/val'
    train_subset = 10000  # Number of training images to use
    val_subset = 1000     # Number of validation images to use
    image_size = 256      # Image resolution
    
    # ==================== Model ====================
    # Parameter count guide (with updated architecture):
    # base_channels=32 → ~6-8M params
    # base_channels=40 → ~10-12M params  
    # base_channels=48 → ~14-16M params
    base_channels = 40    # Use 40 for ~10M params
    
    # Attention configuration
    use_true_attention = False  # Use true attention (slower but more accurate)
    attn_resolution = 32        # Resolution for attention computation (if use_true_attention=True)
    
    # ==================== Training ====================
    epochs = 60  # 改为60，足以收敛且避免过拟合
    batch_size = 8
    lr = 2e-4
    weight_decay = 1e-4
    num_workers = 4
    
    # ==================== Loss Weights ====================
    # Standard configuration
    lambda_coarse = 1.0   # Coarse stage loss
    lambda_refine = 1.0   # Refinement stage loss
    lambda_edge = 0.5     # Edge consistency loss
    lambda_color = 0.5    # Color consistency loss
    lambda_corner = 0.3   # Corner-aware loss
    boundary_width = 5    # Boundary width for color loss (pixels)
    corner_threshold = 0.2  # Corner detection threshold
    
    # ==================== MSSE Configuration ====================
    # MultiScaleStructureEncoder for structure-aware attention bias
    use_structure_encoder = True      # Enable MSSE module
    msse_patch_sizes = (3, 7, 15)     # Multi-scale patch sizes
    msse_boundary_width = 3           # Boundary width for structure encoder (pixels)
    msse_fusion_strength = 0.1        # Structure bias strength in attention (0.05-0.2)
    
    # ==================== Checkpointing ====================
    save_dir = './experiments/exp2'
    resume = None         # Path to checkpoint to resume from
    val_interval = 1      # Validate every N epochs
    save_interval = 10    # Save checkpoint every N epochs
    log_interval = 50     # Log to tensorboard every N batches
    
    # ==================== Metrics ====================
    use_lpips = False     # Use LPIPS metric (requires lpips package)
    
    def __str__(self):
        """Print configuration"""
        config_str = "\n" + "="*60 + "\n"
        config_str += "Configuration:\n"
        config_str += "="*60 + "\n"
        
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key:25s}: {value}\n"
        
        config_str += "="*60 + "\n"
        return config_str


class ConfigHighQuality(Config):
    """Configuration for high quality results (slower training)"""
    
    # More training epochs
    epochs = 100
    
    # Stronger detail losses
    lambda_edge = 1.0
    lambda_color = 1.0
    lambda_corner = 0.5
    
    # MSSE settings - enhanced
    use_structure_encoder = True
    msse_fusion_strength = 0.15  # Slightly stronger
    
    # Use LPIPS
    use_lpips = True
    
    # Larger batch size if GPU allows
    batch_size = 16


class ConfigFastTraining(Config):
    """Configuration for fast training (lower quality)"""
    
    # Fewer epochs
    epochs = 30
    
    # Smaller model
    base_channels = 24  # ~6M params
    
    # Larger batch size
    batch_size = 16
    
    # Less frequent validation
    val_interval = 2
    
    # MSSE disabled for speed
    use_structure_encoder = False


class ConfigLargeModel(Config):
    """Configuration for larger model (better quality, more memory)"""
    
    # Larger model
    base_channels = 40  # ~15M params
    
    # Smaller batch size due to memory
    batch_size = 4
    
    # Stronger detail losses
    lambda_refine = 1.5
    lambda_edge = 1.0
    lambda_color = 1.0
    lambda_corner = 0.5
    
    # MSSE enabled
    use_structure_encoder = True


class ConfigSmallGPU(Config):
    """Configuration for small GPU (< 8GB VRAM)"""
    
    # Reduce memory usage
    batch_size = 4
    base_channels = 24
    num_workers = 2
    
    # MSSE disabled to save memory
    use_structure_encoder = False
    
    # Reduce image size if needed
    # image_size = 128  # Uncomment if still OOM


class ConfigDebug(Config):
    """Debug configuration for testing"""
    
    # Small dataset
    train_subset = 100
    val_subset = 50
    
    # Small model
    base_channels = 16
    
    # Fast training
    epochs = 5
    batch_size = 2
    num_workers = 0
    
    # Logging
    log_interval = 5
    
    # Enable all features for testing
    use_structure_encoder = True
    use_lpips = False


def get_config(name='default'):
    """Get configuration by name"""
    configs = {
        'default': Config,
        'high_quality': ConfigHighQuality,
        'fast': ConfigFastTraining,
        'large': ConfigLargeModel,
        'small_gpu': ConfigSmallGPU,
        'debug': ConfigDebug
    }
    
    if name not in configs:
        print(f"Warning: Config '{name}' not found, using default")
        name = 'default'
    
    return configs[name]()


if __name__ == '__main__':
    # Test configurations
    print("Available configurations:")
    print("1. default (60 epochs, with MSSE)")
    print("2. high_quality (100 epochs, with MSSE)")
    print("3. fast (30 epochs, without MSSE)")
    print("4. large (large model, with MSSE)")
    print("5. small_gpu (memory-efficient, without MSSE)")
    print("6. debug (quick testing)")
    
    print("\n" + "="*60)
    print("Default Configuration:")
    print(Config())
    
    print("\n" + "="*60)
    print("High Quality Configuration:")
    print(ConfigHighQuality())
    
    print("\nMSSE Parameters:")
    print(f"  use_structure_encoder: {Config.use_structure_encoder}")
    print(f"  msse_patch_sizes: {Config.msse_patch_sizes}")
    print(f"  msse_boundary_width: {Config.msse_boundary_width}")
    print(f"  msse_fusion_strength: {Config.msse_fusion_strength}")