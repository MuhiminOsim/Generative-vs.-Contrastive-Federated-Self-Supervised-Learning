"""Federated SSL pre-training script."""

import torch
import yaml
import argparse
import os
import copy
from data.dataset_kaggle import get_kaggle_dataset
from data.dataset_rsna import get_rsna_dataset
from data.transform import get_transform
from models.ssl_models import FedConSwin, FedMaeSwin
from federated.server import FederatedServer
from federated.client import FedConClient, FedMAEClient


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("--- Configuration ---")
    print(config)
    print("---------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model
    if config['ssl_method'] == 'fedcon':
        global_model = FedConSwin(
            model_name=config['model_name'],
            embed_dim=config.get('embed_dim', 768),
            projection_dim=config.get('projection_dim', 128),
            projection_hidden_dim=config.get('projection_hidden_dim', 2048)
        ).to(device)
    elif config['ssl_method'] == 'fedmae':
        global_model = FedMaeSwin(
            model_name=config['model_name'],
            img_size=config.get('img_size', 256),
            patch_size=config.get('patch_size', 4),
            embed_dim=config.get('embed_dim', 768),
            decoder_dim=config.get('decoder_dim', 512),
            mask_ratio=config.get('mask_ratio', 0.75)
        ).to(device)
    else:
        raise ValueError(f"Unknown SSL method: {config['ssl_method']}")

    # 2. Initialize Server
    server = FederatedServer(global_model, config)

    # 3. Initialize Clients and Datasets
    clients = []
    print("Loading client datasets...")
    
    for i in range(config['num_clients']):
        client_id = i + 1
        client_config = config['clients_data'][f'client_{client_id}']
        data_name = client_config['name']
        data_path = client_config['path']
        
        transform = get_transform(
            method=config['ssl_method'],
            image_size=config.get('image_size', 256),
            train=True
        )

        if data_name == 'kaggle':
            dataset = get_kaggle_dataset(
                root_dir=data_path, 
                transform=transform,
                ssl_method=config['ssl_method']
            )
        elif data_name == 'rsna':
            csv_path = client_config.get('csv_path')
            dataset = get_rsna_dataset(
                image_dir=data_path, 
                csv_path=csv_path, 
                transform=transform,
                ssl_method=config['ssl_method']
            )
        else:
            raise ValueError(f"Unknown dataset: {data_name}")
        
        # Create appropriate client type
        if config['ssl_method'] == 'fedcon':
            client = FedConClient(client_id, copy.deepcopy(global_model), config, dataset)
        elif config['ssl_method'] == 'fedmae':
            client = FedMAEClient(client_id, copy.deepcopy(global_model), config, dataset)
        else:
            raise ValueError(f"Unknown SSL method: {config['ssl_method']}")
            
        clients.append(client)
        print(f"Client {client_id} ({data_name}) loaded with {len(dataset)} images.")

    # 4. Start Federated Pre-training
    print("\n--- Starting Federated Pre-training ---")
    for round_num in range(config['num_rounds']):
        print(f"\nCommunication Round {round_num + 1}/{config['num_rounds']}")
        
        # Get global model weights
        global_weights = server.get_global_weights()
        
        client_weights = []
        client_samples = []
        
        # Client local training
        for client in clients:
            client.set_global_weights(global_weights)
            avg_loss = client.train_one_round()
            print(f"  Client {client.client_id} avg loss: {avg_loss:.4f}")
            
            client_weights.append(client.get_local_weights())
            client_samples.append(client.num_samples)
        
        # Server aggregation
        server.aggregate_weights(client_weights, client_samples)
        
        # Save checkpoint periodically
        if (round_num + 1) % config.get('save_every', 10) == 0:
            checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{config['experiment_name']}_round{round_num+1}.pth"
            )
            server.save_encoder(checkpoint_path)
    
    # 5. Save final global model
    output_dir = config.get('output_dir', 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{config['experiment_name']}_final.pth")
    
    server.save_encoder(save_path)
    print(f"\n--- Pre-training complete ---")
    print(f"Final encoder model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated SSL Pre-training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    main(args.config)