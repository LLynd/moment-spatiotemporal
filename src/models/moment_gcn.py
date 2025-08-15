import torch
import torch.nn as nn
import torch.nn.functional as F
from momentfm import MOMENTPipeline

class ChannelGCN(nn.Module):
    def __init__(self, patch_len=8, hidden_dim=128, emb_dim=1024, num_channels=2):
        super().__init__()
        self.num_channels = num_channels
        self.node_proj = nn.Sequential(
            nn.Linear(patch_len, hidden_dim),
            nn.ReLU()
        )
        self.adjacency = nn.Parameter(torch.eye(num_channels) + 0.1 * torch.randn(num_channels, num_channels))
        self.gcn = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, x):
        x = self.node_proj(x)
        adj = F.softmax(self.adjacency, dim=-1)
        x = torch.einsum('ij,bjk->bik', adj, x)
        return self.gcn(x)

class FusionProjection(nn.Module):
    def __init__(self, input_dim=2048, output_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.proj(x)

class ReconstructionMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class MOMENTGCNReconstructor(nn.Module):
    def __init__(
        self,
        pretrained_model: str = "AutonLab/MOMENT-1-large",
        patch_len: int = 8,
        gcn_hidden_dim: int = 128,
        gcn_emb_dim: int = 1024,
        fusion_dim: int = 1024,
        freeze_moment: bool = True
    ):
        """
        Initialize the MOMENT-GCN reconstructor.
        
        Args:
            pretrained_model: Name of pretrained MOMENT model
            patch_len: Length of time series patches
            gcn_hidden_dim: Hidden dimension for GCN node projection
            gcn_emb_dim: Output embedding dimension for GCN
            fusion_dim: Dimension after fusing MOMENT and GCN embeddings
            freeze_moment: Whether to freeze MOMENT weights
        """
        super().__init__()
        self.patch_len = patch_len
        self.freeze_moment = freeze_moment
        
        # Load MOMENT pipeline
        self.moment_model = MOMENTPipeline.from_pretrained(
            pretrained_model,
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': 2
            }
        )
        self.moment_model.init()
        # Extract the actual model from the pipeline
        #self.moment_model = self.moment_pipeline.model
        
        # Get embedding dimension from MOMENT
        self.moment_emb_dim = self.moment_model.config.d_model
        
        if freeze_moment:
            for param in self.moment_model.parameters():
                param.requires_grad = False
                
        # GCN components
        self.gcn = ChannelGCN(
            patch_len=patch_len,
            hidden_dim=gcn_hidden_dim,
            emb_dim=gcn_emb_dim,
            num_channels=2
        )
        
        self.fusion = FusionProjection(
            input_dim=self.moment_emb_dim + gcn_emb_dim,
            output_dim=fusion_dim
        )
        
        self.recon_head = ReconstructionMLP(
            input_dim=fusion_dim,
            output_dim=patch_len
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_len)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        # Normalize input using MOMENT's RevIN
        normalized_x = x#self.moment_model.normalize(x)
        
        # Create patches: (batch_size, num_channels, num_patches, patch_len)
        patches = normalized_x.unfold(-1, self.patch_len, self.patch_len)
        batch_size, num_channels, num_patches, _ = patches.shape
        
        # Get MOMENT embeddings
        with torch.set_grad_enabled(not self.freeze_moment):
            moment_emb = self.moment_model(x)  # (batch_size, num_channels, num_patches, emb_dim)
        
        # Prepare GCN input: group by patch index
        gcn_input = patches.permute(0, 2, 1, 3)  # (batch_size, num_patches, num_channels, patch_len)
        gcn_input = gcn_input.reshape(-1, num_channels, self.patch_len)  # (batch_size*num_patches, num_channels, patch_len)
        
        # Process through GCN
        gcn_emb = self.gcn(gcn_input)  # (batch_size*num_patches, num_channels, gcn_emb_dim)
        
        # Reshape GCN embeddings to match MOMENT format
        gcn_emb = gcn_emb.view(batch_size, num_patches, num_channels, -1)
        gcn_emb = gcn_emb.permute(0, 2, 1, 3)  # (batch_size, num_channels, num_patches, gcn_emb_dim)
        
        # Combine MOMENT and GCN embeddings
        combined = torch.cat([moment_emb, gcn_emb], dim=-1)  # (batch_size, num_channels, num_patches, moment_emb_dim + gcn_emb_dim)
        
        # Project to fusion dimension
        fused_emb = self.fusion(combined)  # (batch_size, num_channels, num_patches, fusion_dim)
        
        # Reconstruct patches
        recon_patches = self.recon_head(
            fused_emb.reshape(-1, fused_emb.size(-1)))  # (batch_size*num_channels*num_patches, patch_len)
        recon_patches = recon_patches.view(batch_size, num_channels, num_patches, self.patch_len)
        
        # Reconstruct time series from patches
        recon_ts = recon_patches.permute(0, 1, 3, 2)  # (batch_size, num_channels, patch_len, num_patches)
        recon_ts = recon_ts.reshape(batch_size, num_channels, -1)  # (batch_size, num_channels, seq_len)
        
        # Denormalize using MOMENT's RevIN
        return self.moment_model.denormalize(recon_ts)