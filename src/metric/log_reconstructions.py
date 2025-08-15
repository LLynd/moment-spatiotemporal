import torch
import wandb
import matplotlib.pyplot as plt

from .base import BaseMetric


class ReconstructionPlotMetric(BaseMetric):
    """Plots original vs reconstructed time series including phase portraits"""
    
    def __init__(self, num_samples=5):
        super().__init__()
        self.originals = []
        self.reconstructions = []
        self.num_samples = num_samples
        
    def forward(self, batch_input, batch_output, batch_misc):
        # Store samples for later plotting
        self.originals.append(batch_input.detach().cpu())
        self.reconstructions.append(batch_output.detach().cpu())
        
    def compute_and_log(self, fabric, log_prefix=""):
        # Only process on rank 0 to avoid duplication
        if not fabric.is_global_zero:
            return
            
        import matplotlib.pyplot as plt
        
        # Concatenate all batches
        originals = torch.cat(self.originals)
        reconstructions = torch.cat(self.reconstructions)
        
        # Create plots for random samples
        figures = []
        indices = torch.randperm(len(originals))[:self.num_samples]
        
        for idx in indices:
            fig = plt.figure(figsize=(14, 8))
            orig = originals[idx]
            recon = reconstructions[idx]
            num_channels = orig.shape[0]
            
            # Original time series
            ax1 = plt.subplot(2, 2, 1)
            for c in range(num_channels):
                ax1.plot(orig[c], label=f'Channel {c+1}')
            ax1.set_title(f"Original Time Series")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Value")
            ax1.legend()
            ax1.grid(True)
            
            # Reconstruction time series
            ax2 = plt.subplot(2, 2, 2)
            for c in range(num_channels):
                ax2.plot(recon[c], label=f'Channel {c+1} Recon')
            ax2.set_title(f"Reconstructed Time Series")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Value")
            ax2.legend()
            ax2.grid(True)
            
            # Phase portrait (only for 2-channel data)
            if num_channels == 2:
                ax3 = plt.subplot(2, 1, 2)
                ax3.plot(orig[0], orig[1], 'b-', label='Original Trajectory')
                ax3.plot(recon[0], recon[1], 'r--', label='Reconstructed Trajectory')
                ax3.set_title("Phase Portrait (Channel 1 vs Channel 2)")
                ax3.set_xlabel("Channel 1")
                ax3.set_ylabel("Channel 2")
                ax3.legend()
                ax3.grid(True)
                ax3.set_aspect('equal', 'box')
            
            plt.tight_layout()
            figures.append(fig)
            plt.close(fig)
        
        # Log to WandB
        fabric.logger.log({
            f"{log_prefix}reconstructions": [wandb.Image(fig) for fig in figures]
        })
        
        # Reset storage
        self.originals = []
        self.reconstructions = []
