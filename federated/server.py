"""Federated learning server for aggregation."""

import torch
import copy


class FederatedServer:
    """Server for federated learning with FedAvg/FedProx aggregation."""
    
    def __init__(self, global_model, config):
        self.model = global_model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_global_weights(self):
        """Return current global model weights."""
        return copy.deepcopy(self.model.state_dict())
    
    def aggregate_weights(self, client_weights_list, client_samples_list):
        """Aggregate client weights using FedAvg (weighted by sample count).
        
        Args:
            client_weights_list: List of state dicts from clients
            client_samples_list: List of sample counts from each client
        """
        # Calculate total samples
        total_samples = sum(client_samples_list)
        
        # Initialize aggregated weights
        aggregated_weights = copy.deepcopy(client_weights_list[0])
        
        # Zero out the weights
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)
        
        # Weighted average
        for client_weights, num_samples in zip(client_weights_list, client_samples_list):
            weight = num_samples / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += client_weights[key].cpu() * weight
        
        # Update global model
        self.model.load_state_dict(aggregated_weights)
    
    def save_model(self, path):
        """Save global model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def save_encoder(self, path):
        """Save only the encoder weights (for downstream tasks)."""
        if hasattr(self.model, 'encoder'):
            torch.save(self.model.encoder.state_dict(), path)
        elif hasattr(self.model, 'get_encoder'):
            torch.save(self.model.get_encoder().state_dict(), path)
        else:
            # Fallback: save entire model
            torch.save(self.model.state_dict(), path)
        print(f"Encoder saved to {path}")
