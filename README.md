# Generative vs. Contrastive Federated Self-Supervised Learning

Official implementation of **Federated Self-Supervised Learning for Medical Imaging** comparing generative (FedMAE) and contrastive (FedCon) approaches using Swin Transformer V2.

## Overview

This repository implements federated self-supervised learning (SSL) methods for pneumonia detection in chest X-rays across heterogeneous datasets (Kaggle Pneumonia, RSNA).

### Methods Implemented

- **FedConSwin**: Federated contrastive learning with NT-Xent loss
- **FedMaeSwin**: Federated masked autoencoder learning
- **FedAvg/FedProx**: Aggregation strategies for heterogeneous clients

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Generative-vs-Contrastive-Federated-SSL.git
cd Generative-vs-Contrastive-Federated-SSL

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

Download and organize datasets:

### Kaggle Pneumonia Dataset
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### RSNA Pneumonia Detection
```
RSNA/
├── Training/
│   └── *.dcm files
└── stage2_train_metadata.csv
```

## Usage

### 1. Pre-training

#### Federated Contrastive (FedConSwin)
```bash
python pretrain.py --config configs/pretrain_fedcon.yaml
```

#### Federated Generative (FedMaeSwin)
```bash
python pretrain.py --config configs/pretrain_fedmae.yaml
```

#### Centralized Contrastive Baseline (CenConSwin)
```bash
python pretrain_centralized.py --config configs/pretrain_cencon.yaml
```

### 2. Fine-tuning

#### From Pre-trained Model (Deep Fine-tuning)
```bash
# Kaggle dataset
python finetune.py --config configs/finetune_kaggle.yaml

# RSNA dataset
python finetune.py --config configs/finetune_rsna.yaml
```

#### From Scratch (Supervised Baseline)
```bash
# Kaggle dataset
python finetune.py --config configs/finetune_scratch.yaml

# RSNA dataset
python finetune.py --config configs/finetune_scratch_rsna.yaml
```

## Configuration

All experiments are configured via YAML files in `configs/`:

**Pre-training:**
- `pretrain_fedcon.yaml`: Federated contrastive SSL (FedConSwin)
- `pretrain_fedmae.yaml`: Federated generative SSL (FedMaeSwin)
- `pretrain_cencon.yaml`: Centralized contrastive baseline (CenConSwin)

**Fine-tuning:**
- `finetune_kaggle.yaml`: Fine-tune on Kaggle dataset
- `finetune_rsna.yaml`: Fine-tune on RSNA dataset
- `finetune_scratch.yaml`: Train from scratch on Kaggle (supervised baseline)
- `finetune_scratch_rsna.yaml`: Train from scratch on RSNA (supervised baseline)

### Key Parameters

**Federated Learning:**
- `num_clients`: Number of federated clients
- `num_rounds`: Communication rounds
- `local_epochs`: Local training epochs per round
- `agg_method`: Aggregation method (`fedavg` or `fedprox`)

**SSL Methods:**
- FedCon: `projection_dim`, `projection_hidden_dim`, `temperature`
- FedMAE: `mask_ratio`, `decoder_dim`

**Fine-tuning Modes:**
- `num_unfrozen_stages`: 
  - `-1`: All stages unfrozen (training from scratch)
  - `0`: All stages frozen (linear probing)
  - `1-4`: Unfreeze last N stages (deep fine-tuning)
- `encoder_learning_rate`: Learning rate for encoder (1e-5 for fine-tuning, 1e-4 for scratch)
- `classifier_learning_rate`: Learning rate for classifier head (1e-3)

## Project Structure

```
.
├── configs/               # YAML configuration files
├── data/                  # Dataset loaders and transforms
│   ├── dataset_kaggle.py
│   ├── dataset_rsna.py
│   └── transform.py
├── models/                # Model definitions
│   ├── ssl_models.py      # FedConSwin, FedMaeSwin
│   └── swin_transformer_v2.py
├── federated/             # Federated learning components
│   ├── client.py          # FedConClient, FedMAEClient
│   └── server.py          # FederatedServer
├── pretrain.py            # Federated pretraining script
├── pretrain_centralized.py # Centralized pretraining (baseline)
├── finetune.py            # Downstream finetuning script
└── requirements.txt       # Dependencies
```

## Experimental Protocol

This repository implements the complete experimental protocol from the paper:

### Pre-training Methods
1. **FedConSwin**: Federated contrastive learning (20 rounds, FedProx aggregation)
2. **FedMaeSwin**: Federated masked autoencoding (20 rounds, FedProx aggregation)  
3. **CenConSwin**: Centralized contrastive baseline (30 epochs, combined data)

### Fine-tuning Strategies
1. **Deep Fine-tuning**: Unfreeze last 2 stages with differential learning rates
2. **Linear Probing**: Freeze all encoder stages (num_unfrozen_stages=0)
3. **Training from Scratch**: Random initialization, full training (num_unfrozen_stages=-1)

## Results

Pre-trained and fine-tuned models will be saved in `checkpoints/`:
- `fedcon_pretrain_final.pth`: FedConSwin encoder weights
- `fedmae_pretrain_final.pth`: FedMaeSwin encoder weights
- `centralized_contrastive_model_final.pth`: CenConSwin encoder weights
- `final_finetuned_model.pth`: Fine-tuned Kaggle model
- `final_finetuned_rsna_model.pth`: Fine-tuned RSNA model
- `final_scratch_model.pth`: Trained-from-scratch baseline

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2025,
  title={Generative vs. Contrastive Federated Self-Supervised Learning for Medical Imaging},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
``` -->

## License

MIT License

## Acknowledgments

- Swin Transformer V2 implementation from [timm](https://github.com/huggingface/pytorch-image-models)
- Federated learning framework using [Flower](https://flower.dev/)
