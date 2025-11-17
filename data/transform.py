"""Data transforms for SSL and finetuning."""

from torchvision.transforms import v2
import torch


def get_transform(method="fedcon", image_size=256, train=True):
    """Get appropriate transforms based on SSL method and phase.
    
    Args:
        method: "fedcon" (contrastive), "fedmae" (MAE), or "finetune"
        image_size: Image size for transforms
        train: Whether to use training or validation transforms
        
    Returns:
        Transform callable
    """
    if method == "fedcon":
        if train:
            return v2.Compose([
                v2.RandomResizedCrop(size=image_size, scale=(0.5, 1.0), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([
                    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
            ])
        else:
            return v2.Compose([
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
            ])
    
    elif method == "fedmae":
        # MAE: simpler augmentation (masking done in model)
        # Using ImageNet normalization like reference
        return v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif method == "finetune":
        if train:
            # Aggressive augmentation for finetuning
            return v2.Compose([
                v2.Resize((image_size, image_size), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([
                    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
                ], p=0.8),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
            ])
        else:
            # Validation: simple resize and normalize
            return v2.Compose([
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
            ])
    
    else:
        raise ValueError(f"Unknown transform method: {method}")


def get_test_transform(image_size=256):
    """Get test/inference transforms (same as eval transforms)."""
    return get_transform(method="finetune", image_size=image_size, train=False)