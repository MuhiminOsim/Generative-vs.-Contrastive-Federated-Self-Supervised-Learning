"""SSL model definitions: FedConSwin (contrastive) and FedMaeSwin (MAE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer_v2 import build_swin_encoder


class FedConSwin(nn.Module):
    """Contrastive learning model with Swin V2 backbone (matches ContrastiveViT from reference)."""
    
    def __init__(self, model_name="swinv2_tiny_window8_256", embed_dim=768, projection_dim=128, projection_hidden_dim=2048):
        super().__init__()
        self.encoder = build_swin_encoder(model_name, pretrained=False)
        
        # Projection head: 2-layer MLP with hidden dimension (paper uses 2048)
        # Architecture: encoder_features -> projection_hidden_dim -> projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.num_features, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim)
        )
        
        self.feature_dim = self.encoder.num_features
        self.projection_dim = projection_dim
        self.projection_hidden_dim = projection_hidden_dim
    
    def forward(self, x):
        """Forward pass - returns projected features."""
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections
    
    def get_encoder(self):
        """Get encoder for downstream tasks."""
        return self.encoder


class FedMaeSwin(nn.Module):
    """Masked Autoencoder with Swin V2 encoder (matches SwinMAE from reference)."""
    
    def __init__(self, model_name="swinv2_tiny_window8_256", img_size=256, patch_size=4, embed_dim=768, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = build_swin_encoder(model_name, pretrained=False)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio
        
        # Simple decoder: 2-layer MLP to reconstruct image
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.num_features, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_dim, img_size * img_size * 3)
        )
        
        self.feature_dim = self.encoder.num_features
    
    def forward(self, x, mask_ratio=None):
        """Forward pass - returns reconstructed image and mask."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input must be {self.img_size}x{self.img_size}"
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        patch_dim = C * self.patch_size * self.patch_size
        
        # Patchify for mask generation
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, patch_dim)
        
        # Create random mask per sample
        mask = torch.rand(B, num_patches, device=x.device) < mask_ratio
        
        # Pass full image to encoder (Swin expects full image)
        encoded = self.encoder(x)
        
        # Decode to reconstruct image
        recon = self.decoder(encoded)
        recon = recon.view(B, 3, self.img_size, self.img_size)
        
        return recon, mask
    
    def get_encoder(self):
        """Get encoder for downstream tasks."""
        return self.encoder


class SwinContrastiveClassifier(nn.Module):
    """Downstream classifier using contrastive encoder (matches ViTClassifier from reference)."""
    
    def __init__(self, encoder, num_classes=1, freeze_encoder=True, num_unfrozen_stages=1):
        super().__init__()
        self.encoder = encoder
        
        # Handle different unfrozen stage configurations:
        # -1: All stages unfrozen (training from scratch)
        # 0: All stages frozen (linear probing)
        # N>0: Unfreeze last N stages (deep fine-tuning)
        
        if num_unfrozen_stages == -1:
            # Training from scratch - all parameters trainable
            freeze_encoder = False
        
        # Freeze all parameters initially
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze the last N stages of the Swin Transformer
            if num_unfrozen_stages > 0:
                if hasattr(self.encoder, 'layers'):
                    num_stages = len(self.encoder.layers)
                    for i in range(num_stages - num_unfrozen_stages, num_stages):
                        for param in self.encoder.layers[i].parameters():
                            param.requires_grad = True
            
            # Always unfreeze the final normalization layer when doing partial unfreezing
            if hasattr(self.encoder, 'norm') and num_unfrozen_stages > 0:
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True
        
        # Classification head (matches reference)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.encoder.num_features),
            nn.Linear(self.encoder.num_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass - returns class logits."""
        features = self.encoder(x)
        logits = self.classifier_head(features)
        return logits


def nt_xent_loss(out1, out2, temperature=0.5):
    """NT-Xent loss matching reference implementation.
    
    Args:
        out1, out2: Projected features from two views [B, D]
        temperature: Temperature parameter for scaling
        
    Returns:
        Loss value
    """
    # Normalize features
    out1 = F.normalize(out1, dim=1)
    out2 = F.normalize(out2, dim=1)
    
    # Concatenate both views
    out = torch.cat([out1, out2], dim=0)
    
    # Compute similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    
    # Mask out diagonal (self-similarity)
    mask = ~torch.eye(out.shape[0], device=sim.device).bool()
    neg = sim.masked_select(mask).view(out.shape[0], -1).sum(dim=-1)
    
    # Positive pairs
    pos = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    
    # Loss
    loss = -torch.log(pos / neg).mean()
    return loss


def mae_loss(reconstruction, target, mask=None):
    """MAE reconstruction loss matching reference implementation.
    
    Args:
        reconstruction: Reconstructed image [B, C, H, W]
        target: Target image [B, C, H, W]
        mask: Optional mask (not used in simplified version)
        
    Returns:
        Loss value
    """
    # Simple mean squared error over all pixels
    loss = ((reconstruction - target) ** 2).mean()
    return loss
