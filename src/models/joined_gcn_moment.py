

class SpatialGNN(nn.Module):
    """Graph Neural Network for spatial relationships within patches"""
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32):
        super().__init__()
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.deconv1 = GCNConv(latent_dim, hidden_dim)
        self.deconv2 = GCNConv(hidden_dim, input_dim)
        
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
    
    def decode(self, z, edge_index):
        z = F.relu(self.deconv1(z, edge_index))
        z = self.deconv2(z, edge_index)
        return z
    
    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        recon = self.decode(z, data.edge_index)
        return z, recon

class PatchReconstructor(nn.Module):
    """Combines MOMENT and GNN embeddings for patch reconstruction"""
    def __init__(self, moment, gnn, moment_dim=512, gnn_dim=32, patch_size=10):
        super().__init__()
        self.moment = moment
        self.gnn = gnn
        self.patch_size = patch_size
        
        # Reconstruction MLP
        self.mlp_x = nn.Sequential(
            nn.Linear(moment_dim + gnn_dim, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size)
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(moment_dim + gnn_dim, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size)
        )
        
    def forward(self, moment_patch, graph_data):
        # Get MOMENT embedding (frozen)
        with torch.no_grad():
            moment_embed = self.moment.encoder(moment_patch)
            moment_embed = moment_embed.mean(dim=1)  # [batch, moment_dim]
        
        # Get GNN embedding
        gnn_embed, _ = self.gnn(graph_data)  # [batch * patch_size, gnn_dim]
        
        # Average over time points to get patch-level embedding
        gnn_embed = gnn_embed.view(-1, self.patch_size, gnn_embed.size(-1))
        gnn_embed = gnn_embed.mean(dim=1)  # [batch, gnn_dim]
        
        # Combine embeddings
        combined_x = torch.cat([moment_embed, gnn_embed], dim=1)
        combined_y = torch.cat([moment_embed, gnn_embed], dim=1)
        
        # Reconstruct channels
        recon_x = self.mlp_x(combined_x)
        recon_y = self.mlp_y(combined_y)
        
        return recon_x, recon_y
