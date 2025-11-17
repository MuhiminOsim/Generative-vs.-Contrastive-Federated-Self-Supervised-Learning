"""Federated learning clients for FedCon and FedMAE."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import copy
from models.ssl_models import nt_xent_loss, mae_loss


class BaseClient:
    """Base client for federated learning."""
    
    def __init__(self, client_id, model, config, train_dataset):
        self.client_id = client_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        self.num_samples = len(train_dataset)
    
    def set_global_weights(self, global_weights):
        """Update local model with global weights."""
        self.model.load_state_dict(global_weights)
    
    def get_local_weights(self):
        """Return local model weights."""
        return self.model.state_dict()
    
    def train_one_round(self):
        """Train for one federated round - to be implemented by subclasses."""
        raise NotImplementedError


class FedConClient(BaseClient):
    """Client for FedCon (contrastive learning)."""
    
    def train_one_round(self):
        """Train contrastive model for local epochs."""
        self.model.train()
        
        # Store global weights for FedProx
        global_weights = None
        if self.config.get('agg_method') == 'fedprox':
            global_weights = copy.deepcopy(self.model.state_dict())
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1.5e-4),
            weight_decay=self.config.get('weight_decay', 0.05)
        )
        
        # Mixed precision training
        gradScaler = GradScaler(enabled=torch.cuda.is_available())
        
        epoch_losses = []
        local_epochs = self.config.get('local_epochs', 2)
        
        for epoch in range(local_epochs):
            batch_losses = []
            
            for batch in self.train_loader:
                # Batch contains two views: (view1, view2)
                view1, view2 = batch
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=torch.cuda.is_available()):
                    proj1 = self.model(view1)
                    proj2 = self.model(view2)
                    loss = nt_xent_loss(proj1, proj2, temperature=self.config.get('temperature', 0.5))
                    
                    # FedProx regularization
                    if global_weights is not None:
                        prox_loss = 0.0
                        for name, param in self.model.named_parameters():
                            prox_loss += ((param - global_weights[name].to(self.device)) ** 2).sum()
                        loss += (self.config.get('mu', 0.01) / 2) * prox_loss
                
                gradScaler.scale(loss).backward()
                gradScaler.step(optimizer)
                gradScaler.update()
                
                batch_losses.append(loss.item())
            
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        
        return sum(epoch_losses) / len(epoch_losses)


class FedMAEClient(BaseClient):
    """Client for FedMAE (masked autoencoder)."""
    
    def train_one_round(self):
        """Train MAE model for local epochs."""
        self.model.train()
        
        # Store global weights for FedProx
        global_weights = None
        if self.config.get('agg_method') == 'fedprox':
            global_weights = copy.deepcopy(self.model.state_dict())
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1.5e-4),
            weight_decay=self.config.get('weight_decay', 0.05)
        )
        
        # Mixed precision training
        gradScaler = GradScaler(enabled=torch.cuda.is_available())
        
        epoch_losses = []
        local_epochs = self.config.get('local_epochs', 2)
        
        for epoch in range(local_epochs):
            batch_losses = []
            
            for batch in self.train_loader:
                # For MAE, dataset returns single images
                if isinstance(batch, (list, tuple)):
                    images = batch[0] if len(batch) > 1 else batch
                else:
                    images = batch
                    
                images = images.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=torch.cuda.is_available()):
                    reconstruction, mask = self.model(images, mask_ratio=self.config.get('mask_ratio', 0.75))
                    loss = mae_loss(reconstruction, images, mask)
                    
                    # FedProx regularization
                    if global_weights is not None:
                        prox_loss = 0.0
                        for name, param in self.model.named_parameters():
                            prox_loss += ((param - global_weights[name].to(self.device)) ** 2).sum()
                        loss += (self.config.get('mu', 0.01) / 2) * prox_loss
                
                gradScaler.scale(loss).backward()
                gradScaler.step(optimizer)
                gradScaler.update()
                
                batch_losses.append(loss.item())
            
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        
        return sum(epoch_losses) / len(epoch_losses)
