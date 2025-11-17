"""Downstream finetuning script for classification."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision.transforms import v2 as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import yaml
import argparse


class ChestXrayDataset(Dataset):
    """Chest X-ray dataset for binary classification."""
    
    def __init__(self, image_dir, transform=None, label=0):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.endswith(('.jpeg', '.jpg', '.png'))
        ]
        self.label = label
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


class RSNAClassificationDataset(Dataset):
    """RSNA dataset for binary classification."""
    
    def __init__(self, image_dir, csv_path, transform=None, split='train'):
        import pandas as pd
        self.image_dir = image_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_path)
        self.split = split
        
        # Filter by patientId to get unique patients
        self.data = self.metadata.groupby('patientId').first().reset_index()
        self.image_files = []
        self.labels = []
        
        for _, row in self.data.iterrows():
            patient_id = row['patientId']
            # Try .png first (preprocessed), then .dcm
            img_path_png = os.path.join(image_dir, f"{patient_id}.png")
            img_path_dcm = os.path.join(image_dir, f"{patient_id}.dcm")
            
            if os.path.exists(img_path_png):
                self.image_files.append(img_path_png)
                self.labels.append(int(row['Target']))
            elif os.path.exists(img_path_dcm):
                self.image_files.append(img_path_dcm)
                self.labels.append(int(row['Target']))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        # Load image based on extension
        if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
            image = Image.open(img_path).convert("RGB")
        else:  # .dcm file
            try:
                import pydicom
                dcm = pydicom.dcmread(img_path)
                image = Image.fromarray(dcm.pixel_array).convert("RGB")
            except ImportError:
                raise ImportError("pydicom required for .dcm files. Install with: pip install pydicom")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def prepare_kaggle_data(data_root, batch_size=32):
    """Prepare balanced Kaggle Pneumonia train and test datasets."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
    ])
    
    # Training data - balance classes
    normal_train_dir = os.path.join(data_root, "train", "NORMAL")
    anomaly_train_dir = os.path.join(data_root, "train", "PNEUMONIA")
    
    normal_train_set = ChestXrayDataset(
        image_dir=normal_train_dir, 
        transform=train_transform, 
        label=0
    )
    anomaly_train_set = ChestXrayDataset(
        image_dir=anomaly_train_dir, 
        transform=train_transform, 
        label=1
    )
    
    # Balance by downsampling majority class
    min_len = min(len(normal_train_set), len(anomaly_train_set))
    normal_subset, _ = random_split(
        normal_train_set, 
        [min_len, len(normal_train_set) - min_len]
    )
    anomaly_subset, _ = random_split(
        anomaly_train_set, 
        [min_len, len(anomaly_train_set) - min_len]
    )
    
    finetune_train_set = ConcatDataset([normal_subset, anomaly_subset])
    finetune_loader = DataLoader(
        finetune_train_set, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Test data
    normal_test_dir = os.path.join(data_root, "test", "NORMAL")
    anomaly_test_dir = os.path.join(data_root, "test", "PNEUMONIA")
    
    normal_test_set = ChestXrayDataset(
        image_dir=normal_test_dir, 
        transform=test_transform, 
        label=0
    )
    anomaly_test_set = ChestXrayDataset(
        image_dir=anomaly_test_dir, 
        transform=test_transform, 
        label=1
    )
    
    test_set = ConcatDataset([normal_test_set, anomaly_test_set])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return finetune_loader, test_loader


def prepare_rsna_data(image_dir, csv_path, batch_size=32, train_split=0.8):
    """Prepare RSNA dataset with train/test split."""
    from sklearn.model_selection import train_test_split
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4799, 0.4799, 0.4799], std=[0.2391, 0.2391, 0.2391])
    ])
    
    # Load full dataset
    full_dataset_train = RSNAClassificationDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=train_transform,
        split='train'
    )
    
    full_dataset_test = RSNAClassificationDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=test_transform,
        split='test'
    )
    
    # Split into train/test
    train_size = int(train_split * len(full_dataset_train))
    test_size = len(full_dataset_train) - train_size
    
    train_dataset, test_dataset_from_train = random_split(
        full_dataset_train,
        [train_size, test_size]
    )
    
    # Update test dataset transform
    test_dataset = torch.utils.data.Subset(full_dataset_test, range(test_size))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset_from_train,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def run_finetuning(config_path):
    """Run the complete finetuning pipeline."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained model
    from models.ssl_models import SwinContrastiveClassifier
    from models.swin_transformer_v2 import build_swin_encoder
    
    # Build encoder and classifier
    encoder = build_swin_encoder(
        config.get('model_name', 'swinv2_tiny_window8_256'),
        pretrained=False
    )
    
    # Load pretrained weights if available
    pretrained_path = config.get('pretrained_model_path')
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained encoder from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Handle FedConSwin model (has encoder.encoder)
        if any('encoder.encoder' in k for k in state_dict.keys()):
            encoder_state = {k.replace('encoder.encoder.', ''): v 
                           for k, v in state_dict.items() 
                           if k.startswith('encoder.encoder.')}
            encoder.load_state_dict(encoder_state, strict=False)
        else:
            encoder.load_state_dict(state_dict, strict=False)
        training_mode = "Fine-tuning from pre-trained model"
    else:
        print("Training from scratch (no pre-trained weights)")
        training_mode = "Training from scratch"
    
    model = SwinContrastiveClassifier(
        encoder=encoder,
        num_classes=1,
        num_unfrozen_stages=config.get('num_unfrozen_stages', 0)
    )
    model.to(device)
    
    # Prepare data based on dataset type
    dataset_type = config.get('dataset', 'kaggle').lower()
    
    if dataset_type == 'kaggle':
        print(f"Loading Kaggle Pneumonia dataset from {config['data_path']}")
        train_loader, test_loader = prepare_kaggle_data(
            config['data_path'],
            batch_size=config.get('batch_size', 32)
        )
    elif dataset_type == 'rsna':
        print(f"Loading RSNA dataset from {config['data_path']}")
        train_loader, test_loader = prepare_rsna_data(
            config['data_path'],
            config['csv_path'],
            batch_size=config.get('batch_size', 32),
            train_split=config.get('train_split', 0.8)
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: 'kaggle', 'rsna'")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    
    # Differential learning rates: encoder (if unfrozen) vs classifier head (paper strategy)
    encoder_lr = config.get('encoder_learning_rate', 1e-5)
    classifier_lr = config.get('classifier_learning_rate', 1e-3)
    
    # Build parameter groups
    param_groups = []
    
    # Add encoder parameters if unfrozen (num_unfrozen_stages > 0 or -1 for all)
    num_unfrozen = config.get('num_unfrozen_stages', 0)
    if num_unfrozen != 0:  # Not zero means some or all encoder params are trainable
        # Get unfrozen encoder parameters (from SwinContrastiveClassifier)
        encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': encoder_lr})
            print(f"Encoder: {len(encoder_params)} trainable parameter tensors with LR={encoder_lr}")
            if num_unfrozen == -1:
                print("  → Training from scratch (all stages unfrozen)")
            else:
                print(f"  → Deep fine-tuning (last {num_unfrozen} stages unfrozen)")
    else:
        print("Encoder: Fully frozen (linear probing)")
    
    # Add classifier head parameters
    classifier_params = model.classifier_head.parameters()
    param_groups.append({'params': classifier_params, 'lr': classifier_lr})
    print(f"Classifier head: LR={classifier_lr}")
    
    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config.get('epochs', 35), 
        eta_min=1e-7
    )
    gradScaler = GradScaler(enabled=torch.cuda.is_available())
    
    print(f"\n--- Starting Supervised Training ---")
    print(f"Mode: {training_mode}")
    print(f"Pretrained model: {pretrained_path if pretrained_path else 'None (training from scratch)'}")
    print(f"Dataset: {dataset_type.upper()} - {config['data_path']}")
    print(f"Epochs: {config.get('epochs', 35)}")
    print(f"Batch size: {config.get('batch_size', 32)}")
    
    # Training loop
    for epoch in range(config.get('epochs', 35)):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            
            gradScaler.scale(loss).backward()
            gradScaler.step(optimizer)
            gradScaler.update()
            
            running_loss += loss.item()
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.get('epochs', 35)}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    output_path = config.get('output_model_path', 'final_finetuned_model.pth')
    torch.save(model.state_dict(), output_path)
    print(f"\n--- Model saved to {output_path} ---")
    
    # Evaluation
    print("\n--- Evaluating Fine-tuned Model ---")
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze().cpu().numpy()
            all_preds.extend(outputs.tolist())
            all_labels.extend(labels)
    
    # Compute metrics
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    
    # AUC
    final_auc = roc_auc_score(all_labels_np, all_preds_np)
    print(f"\nAUC Score on Test Set: {final_auc:.4f}")
    
    # Binary predictions with threshold 0.5
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds_np]
    accuracy = accuracy_score(all_labels_np, binary_preds)
    precision = precision_score(all_labels_np, binary_preds)
    recall = recall_score(all_labels_np, binary_preds)
    f1 = f1_score(all_labels_np, binary_preds)
    
    print(f"Accuracy (threshold=0.5): {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_preds_np)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_j_idx]
    
    optimal_binary_preds = [1 if p >= optimal_threshold else 0 for p in all_preds_np]
    optimal_accuracy = accuracy_score(all_labels_np, optimal_binary_preds)
    optimal_precision = precision_score(all_labels_np, optimal_binary_preds)
    optimal_recall = recall_score(all_labels_np, optimal_binary_preds)
    optimal_f1 = f1_score(all_labels_np, optimal_binary_preds)
    
    print(f"\n--- Optimal Threshold (Youden's J) ---")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {optimal_accuracy:.4f}")
    print(f"Precision: {optimal_precision:.4f}")
    print(f"Recall: {optimal_recall:.4f}")
    print(f"F1-score: {optimal_f1:.4f}")
    
    # 95% confidence interval for AUC using bootstrap
    print(f"\n--- Bootstrap 95% CI for AUC ---")
    n_bootstraps = 1000
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(all_labels_np), len(all_labels_np))
        score = roc_auc_score(all_labels_np[indices], all_preds_np[indices])
        bootstrapped_scores.append(score)
    
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    print(f"95% CI for AUC: [{lower:.4f}, {upper:.4f}]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Finetune pretrained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    run_finetuning(args.config)


if __name__ == "__main__":
    main()
