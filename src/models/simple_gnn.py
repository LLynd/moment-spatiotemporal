import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv


class SimpleGNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16):
        super().__init__()
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

        # Decoder
        self.deconv1 = GCNConv(latent_dim, hidden_dim)
        self.deconv2 = GCNConv(hidden_dim, input_dim)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        z = self.deconv1(z, edge_index)
        z = F.relu(z)
        z = self.deconv2(z, edge_index)
        return z

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        recon = self.decode(z, data.edge_index)
        return z, recon  # Return both embeddings and reconstructions