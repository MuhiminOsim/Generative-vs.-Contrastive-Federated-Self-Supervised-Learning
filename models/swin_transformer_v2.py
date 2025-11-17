"""Swin Transformer encoder from timm."""

import torch
import torch.nn as nn
import timm


def build_swin_encoder(model_name="swinv2_tiny_window8_256", pretrained=False):
    """Build Swin Transformer V2 encoder from timm.
    
    Args:
        model_name: Model name in timm. Options: swinv2_tiny_window8_256, swinv2_small_window8_256, swinv2_base_window8_256
        pretrained: Whether to load ImageNet pretrained weights
        
    Returns:
        Model with feature output (removes classification head)
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    # Remove the classification head to get feature output
    return model


class SwinEncoder(nn.Module):
    """Swin Transformer V2 encoder wrapper."""
    
    def __init__(self, model_name="swinv2_tiny_window8_256", pretrained=False):
        super().__init__()
        self.encoder = build_swin_encoder(model_name, pretrained=pretrained)
        # Get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        self.feature_dim = features.shape[1] if len(features.shape) > 1 else features.shape[-1]
    
    def forward(self, x):
        """Forward pass returns feature vector."""
        return self.encoder(x)
    
    def get_feature_dim(self):
        """Get output feature dimension."""
        return self.feature_dim


def freeze_encoder_stages(model, num_unfrozen_stages=0):
    """Freezes the encoder, leaving the last 'num_unfrozen_stages' unfrozen."""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the head (classifier) if it exists
    if hasattr(model, 'head') and model.head is not None:
        for param in model.head.parameters():
            param.requires_grad = True
        
    # Unfreeze the final normalization layer
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True
            
    # Unfreeze the specified number of final stages
    if hasattr(model, 'layers') and num_unfrozen_stages > 0:
        num_stages = len(model.layers)
        for i in range(num_stages - num_unfrozen_stages, num_stages):
            for param in model.layers[i].parameters():
                param.requires_grad = True
                
    print(f"Model frozen. Unfrozen stages: {num_unfrozen_stages}. Head is unfrozen.")
    return model