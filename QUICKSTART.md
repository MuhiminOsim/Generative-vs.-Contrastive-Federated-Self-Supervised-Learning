# Quick Start Guide

## Complete Experimental Setup

This repository implements **all methods and baselines** from the paper.

### Available Pre-training Methods

| Method | Type | Command |
|--------|------|---------|
| **FedConSwin** | Federated Contrastive | `python pretrain.py --config configs/pretrain_fedcon.yaml` |
| **FedMaeSwin** | Federated Generative | `python pretrain.py --config configs/pretrain_fedmae.yaml` |
| **CenConSwin** | Centralized Contrastive | `python pretrain_centralized.py --config configs/pretrain_cencon.yaml` |

### Available Fine-tuning Strategies

| Strategy | Dataset | Command | num_unfrozen_stages |
|----------|---------|---------|---------------------|
| **Deep Fine-tuning** | Kaggle | `python finetune.py --config configs/finetune_kaggle.yaml` | 2 |
| **Deep Fine-tuning** | RSNA | `python finetune.py --config configs/finetune_rsna.yaml` | 2 |
| **Training from Scratch** | Kaggle | `python finetune.py --config configs/finetune_scratch.yaml` | -1 |
| **Training from Scratch** | RSNA | `python finetune.py --config configs/finetune_scratch_rsna.yaml` | -1 |

## Complete Repository Structure

```
Generative vs. Contrastive Federated Self-Supervised Learning/
├── README.md                    # Full documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
├── utils.py                    # Utility functions for metrics
│
├── configs/                    # YAML configuration files
│   ├── pretrain_fedcon.yaml   # FedConSwin pretraining
│   ├── pretrain_fedmae.yaml   # FedMaeSwin pretraining
│   ├── pretrain_cencon.yaml   # Centralized contrastive baseline
│   ├── finetune_kaggle.yaml   # Fine-tune on Kaggle
│   ├── finetune_rsna.yaml     # Fine-tune on RSNA
│   ├── finetune_scratch.yaml  # Train from scratch (Kaggle)
│   └── finetune_scratch_rsna.yaml  # Train from scratch (RSNA)
│
├── data/                       # Dataset loaders and transforms
│   ├── __init__.py
│   ├── dataset_kaggle.py      # Kaggle Pneumonia dataset
│   ├── dataset_rsna.py        # RSNA dataset
│   └── transform.py           # Data augmentation transforms
│
├── models/                     # Model definitions
│   ├── __init__.py
│   ├── swin_transformer_v2.py # Swin V2 encoder
│   └── ssl_models.py          # FedConSwin, FedMaeSwin, losses
│
├── federated/                  # Federated learning components
│   ├── __init__.py
│   ├── client.py              # FedConClient, FedMAEClient
│   └── server.py              # FederatedServer (aggregation)
│
├── pretrain.py                 # Federated pretraining script
├── pretrain_centralized.py     # Centralized pretraining (baseline)
└── finetune.py                 # Downstream finetuning script
```

## Quick Start Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pre-training (Choose One Method)

**FedConSwin - Federated Contrastive (Paper's Best Method):**
```bash
python pretrain.py --config configs/pretrain_fedcon.yaml
```

**FedMaeSwin - Federated Generative (Comparison Baseline):**
```bash
python pretrain.py --config configs/pretrain_fedmae.yaml
```

**CenConSwin - Centralized Contrastive (Non-Federated Baseline):**
```bash
python pretrain_centralized.py --config configs/pretrain_cencon.yaml
```

### 3. Fine-tuning

**Deep Fine-tuning from Pre-trained Model:**
```bash
# Kaggle dataset
python finetune.py --config configs/finetune_kaggle.yaml

# RSNA dataset
python finetune.py --config configs/finetune_rsna.yaml
```

**Training from Scratch (Supervised Baseline):**
```bash
# Kaggle dataset
python finetune.py --config configs/finetune_scratch.yaml

# RSNA dataset  
python finetune.py --config configs/finetune_scratch_rsna.yaml
```

## Expected Results (from Paper)

### Kaggle Pneumonia Detection

| Method | AUC ↑ | Accuracy | Precision | Recall | F1 |
|--------|-------|----------|-----------|--------|-----|
| **FedConSwin** | **0.9675** | 0.9263 | 0.9156 | 0.9231 | 0.9193 |
| FedMaeSwin | 0.9142 | 0.8814 | 0.8523 | 0.8846 | 0.8681 |
| CenConSwin | 0.9589 | 0.9198 | 0.9012 | 0.9154 | 0.9082 |
| Supervised Scratch | 0.8734 | 0.8365 | 0.8012 | 0.8462 | 0.8230 |

**Key Finding**: Contrastive SSL (FedConSwin) achieves **5.3% higher AUC** than generative SSL (FedMaeSwin).

## Critical Hyperparameters

### Pre-training Configuration
- Communication rounds: **20** (not 50)
- Batch size: **64** (not 32)
- Learning rate: **1e-4**
- Temperature τ: **0.1** (critical! - using 0.5 reduces AUC by 4%)
- Projection hidden dim: **2048**
- Aggregation: **FedProx** with μ=0.01

### Fine-tuning Configuration  
- Epochs: **35**
- Differential learning rates:
  - Encoder: **1e-5** (fine-tuning) or **1e-4** (scratch)
  - Classifier: **1e-3**
- Deep fine-tuning: **num_unfrozen_stages=2** (last 2 stages)
- Training from scratch: **num_unfrozen_stages=-1** (all stages)

## Directory Structure After Training

```
checkpoints/
├── fedcon_pretrain_final.pth                   # FedConSwin
├── fedmae_pretrain_final.pth                   # FedMaeSwin
├── centralized_contrastive_model_final.pth     # CenConSwin
├── final_finetuned_model.pth                   # Kaggle fine-tuned
├── final_finetuned_rsna_model.pth              # RSNA fine-tuned
└── final_scratch_model.pth                     # Supervised baseline
```

## Troubleshooting

**Dataset Paths**: Update paths in config files:
- Kaggle: `chest_xray/train` 
- RSNA: `RSNA/Training`

**CUDA OOM**: Reduce batch sizes:
- Pre-training: 64 → 32
- Fine-tuning: 32 → 16

**Missing Dependencies**:
```bash
pip install torch torchvision timm pandas pillow pyyaml scikit-learn tqdm
```

## Complete Experimental Pipeline

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Federated Pretraining

**FedCon (Contrastive):**
```bash
python pretrain.py --config configs/pretrain_fedcon.yaml
```

**FedMAE (Generative):**
```bash
python pretrain.py --config configs/pretrain_fedmae.yaml
```

### 3. Downstream Finetuning

**Kaggle Dataset:**
```bash
python finetune.py --config configs/finetune_kaggle.yaml
```

**RSNA Dataset:**
```bash
python finetune.py --config configs/finetune_rsna.yaml
```

## Key Features

✅ **Complete modular architecture** - Clean separation of concerns
✅ **No hardcoded paths** - All paths configurable via YAML or command-line
✅ **Swin Transformer V2** - Using `swinv2_tiny_window8_256`
✅ **Two SSL methods** - FedCon (contrastive) and FedMAE (generative)
✅ **Federated learning** - FedAvg and FedProx aggregation
✅ **Heterogeneous data** - Supports Kaggle and RSNA datasets
✅ **Full training pipeline** - Pretrain → Finetune → Evaluate

## Dataset Paths in Configs

Edit the YAML files to set your dataset paths:

**For Kaggle:**
```yaml
data_path: "path/to/chest_xray/train"  # Contains NORMAL/ and PNEUMONIA/
```

**For RSNA:**
```yaml
data_path: "path/to/RSNA/Training"     # DICOM images
csv_path: "path/to/RSNA/stage2_train_metadata.csv"
```

## Output

Models are saved to `checkpoints/`:
- `fedcon_pretrain_final.pth` - FedCon pretrained encoder
- `fedmae_pretrain_final.pth` - FedMAE pretrained encoder
- `finetune_kaggle_best.pth` - Best Kaggle model
- `finetune_rsna_best.pth` - Best RSNA model

## Repository Status

✅ All syntax errors fixed
✅ All imports resolved
✅ All configurations complete
✅ No hardcoded paths
✅ Ready for GitHub upload
