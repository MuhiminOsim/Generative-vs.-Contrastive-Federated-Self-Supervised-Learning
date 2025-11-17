"""Centralized contrastive pre-training (CenConSwin baseline)."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml
import argparse

from models.ssl_models import FedConSwin, nt_xent_loss
from data.dataset_kaggle import KagglePneumoniaDataset
from data.dataset_rsna import RSNADataset
from data.transform import get_transform


def train_centralized_contrastive(config_path):
    """Train a centralized contrastive model (non-federated baseline)."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("--- Centralized Contrastive Pre-training Configuration ---")
    print(config)
    print("-----------------------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = FedConSwin(
        model_name=config['model_name'],
        embed_dim=config.get('embed_dim', 768),
        projection_dim=config.get('projection_dim', 128),
        projection_hidden_dim=config.get('projection_hidden_dim', 2048)
    ).to(device)
    
    # Prepare datasets - combine all client data
    transform = get_transform(method='fedcon', image_size=config.get('image_size', 256), train=True)
    
    datasets = []
    
    # Load Kaggle dataset
    kaggle_path = config['clients_data']['client_1']['path']
    kaggle_dataset = KagglePneumoniaDataset(
        root_dir=kaggle_path,
        transform=transform,
        ssl_method='fedcon'
    )
    datasets.append(kaggle_dataset)
    print(f"Loaded Kaggle dataset: {len(kaggle_dataset)} samples")
    
    # Load RSNA dataset
    rsna_path = config['clients_data']['client_2']['path']
    rsna_csv = config['clients_data']['client_2']['csv_path']
    rsna_dataset = RSNADataset(
        image_dir=rsna_path,
        csv_path=rsna_csv,
        transform=transform,
        ssl_method='fedcon'
    )
    datasets.append(rsna_dataset)
    print(f"Loaded RSNA dataset: {len(rsna_dataset)} samples")
    
    # Combine all data
    combined_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(
        combined_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    total_samples = len(combined_dataset)
    print(f"Total training samples: {total_samples}")
    
    # Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.05)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('num_epochs', 30),
        eta_min=1e-7
    )
    
    gradScaler = GradScaler(enabled=torch.cuda.is_available())
    temperature = config.get('temperature', 0.1)
    
    # Training loop
    print("\n--- Starting Centralized Contrastive Pre-training ---")
    
    for epoch in range(config.get('num_epochs', 30)):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.get('num_epochs', 30)}")
        
        for batch in progress_bar:
            view1, view2 = batch
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                proj1 = model(view1)
                proj2 = model(view2)
                loss = nt_xent_loss(proj1, proj2, temperature=temperature)
            
            gradScaler.scale(loss).backward()
            gradScaler.step(optimizer)
            gradScaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{config.get('num_epochs', 30)}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            checkpoint_path = os.path.join(
                config.get('checkpoint_dir', 'checkpoints'),
                f"centralized_contrastive_model_epoch{epoch+1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(
        config.get('output_dir', 'checkpoints'),
        'centralized_contrastive_model_final.pth'
    )
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\n--- Centralized Pre-training Complete ---")
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Contrastive Pre-training")
    parser.add_argument('--config', type=str, default='configs/pretrain_cencon.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    train_centralized_contrastive(args.config)
