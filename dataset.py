import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2


def random_free_form_mask(h, w, max_strokes=5):
    """Generate random free-form mask"""
    mask = np.zeros((h, w), np.float32)
    for _ in range(random.randint(1, max_strokes)):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(h // 8, h // 3)
        brush_w = random.randint(8, 20)
        for i in range(length):
            x += int(np.cos(angle))
            y += int(np.sin(angle))
            angle += random.uniform(-0.5, 0.5)  # Add curvature
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(mask, (x, y), brush_w, 1, -1)
    return mask


def random_rectangular_mask(h, w, min_size_ratio=0.1, max_size_ratio=0.3):
    """Generate random rectangular mask"""
    mask = np.zeros((h, w), np.float32)
    mask_h = random.randint(int(h * min_size_ratio), int(h * max_size_ratio))
    mask_w = random.randint(int(w * min_size_ratio), int(w * max_size_ratio))
    
    top = random.randint(0, h - mask_h)
    left = random.randint(0, w - mask_w)
    
    mask[top:top+mask_h, left:left+mask_w] = 1
    return mask


def mixed_mask(h, w):
    """Generate mixed mask (rectangular + free-form)"""
    if random.random() < 0.5:
        return random_rectangular_mask(h, w)
    else:
        return random_free_form_mask(h, w)


class InpaintingDataset(Dataset):
    """Dataset for image inpainting"""
    def __init__(self, root, image_size=256, mask_type='mixed', subset_size=None):
        """
        Args:
            root: path to image directory
            image_size: resize images to this size
            mask_type: 'free_form', 'rectangular', or 'mixed'
            subset_size: if not None, randomly select this many images
        """
        self.root = root
        self.image_size = image_size
        self.mask_type = mask_type
        
        # Find all images
        self.images = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
        ])
        
        # Subset selection for faster training
        if subset_size is not None and subset_size < len(self.images):
            random.seed(42)  # For reproducibility
            self.images = random.sample(self.images, subset_size)
            random.seed()  # Reset seed
        
        print(f"Dataset initialized with {len(self.images)} images")
        
        # Image transforms
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)
        
        _, h, w = img.shape
        
        # Generate mask
        if self.mask_type == 'free_form':
            mask = random_free_form_mask(h, w)
        elif self.mask_type == 'rectangular':
            mask = random_rectangular_mask(h, w)
        else:  # mixed
            mask = mixed_mask(h, w)
        
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return {
            "image": img,           # [3, H, W], ground truth
            "mask": mask,           # [1, H, W], 1=missing, 0=valid
            "img_path": img_path
        }


def get_dataloaders(train_root, val_root, batch_size=8, image_size=256, 
                   num_workers=4, train_subset=10000, val_subset=1000):
    """Create train and validation dataloaders"""
    
    train_dataset = InpaintingDataset(
        train_root, 
        image_size=image_size, 
        mask_type='mixed',
        subset_size=train_subset
    )
    
    val_dataset = InpaintingDataset(
        val_root, 
        image_size=image_size, 
        mask_type='mixed',
        subset_size=val_subset
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset = InpaintingDataset("./data/train", image_size=256, subset_size=100)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Mask ratio: {sample['mask'].sum() / sample['mask'].numel():.2%}")