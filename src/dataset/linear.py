import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from dataclasses import dataclass
from typing import Union, Dict, Tuple, Optional


@dataclass
class Batch:
    index: torch.Tensor
    data: torch.Tensor  # (batch_size, 2, seq_len)

def time_series_collate_fn(batch):
    indices = torch.tensor([item[0] for item in batch])
    data = torch.stack([item[1] for item in batch])
    return Batch(indices, data)


class LinearSystemDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        mode: str = "synthetic",
        synthetic_params: Optional[Dict] = None,
        data_path: Optional[Union[str, Dict[str, str]]] = None,
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        Time series dataset for foundation model training.

        Args:
            split: "train" or "test"
            mode: "synthetic" or "file"
            synthetic_params: Parameters for synthetic data generation
            data_path: Path to data file(s):
                - For "file" mode with single file: string path
                - For separate train/test files: dict {"train": path, "test": path}
            train_ratio: Train-test split ratio (for synthetic/single-file mode)
            seed: Random seed for reproducibility
        """
        super().__init__()
        assert split in ["train", "test"], "Split must be 'train' or 'test'"
        assert mode in ["synthetic", "file"], "Mode must be 'synthetic' or 'file'"
        
        self.split = split
        self.mode = mode
        self.seed = seed
        self.data = None
        
        # Set default synthetic parameters
        default_synthetic_params = {
            "num_trajectories": 10000,
            "system_type": "stable_spiral",
            "num_timepoints": 100,
            "t_span": (0, 10),
            "x0_range": (-2, 2)
        }
        
        if mode == "synthetic":
            # Merge user parameters with defaults
            self.synthetic_params = {**default_synthetic_params, **(synthetic_params or {})}
            self._generate_synthetic_data(train_ratio)
            
        elif mode == "file":
            self._load_file_data(data_path, train_ratio)

    def _generate_synthetic_data(self, train_ratio: float):
        """Generate synthetic time series data"""
        # Generate the full dataset
        full_data = self._generate_trajectory_dataset(**self.synthetic_params)
        
        # Convert to tensor (num_samples, 2, seq_len)
        full_data = torch.tensor(full_data, dtype=torch.float32).permute(0, 2, 1)
        
        # Split into train/test
        num_train = int(len(full_data) * train_ratio)
        num_test = len(full_data) - num_train
        
        # Use random_split for reproducible splitting
        train_data, test_data = random_split(
            full_data,
            [num_train, num_test],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        self.data = train_data if self.split == "train" else test_data

    def _load_file_data(self, data_path: Union[str, Dict], train_ratio: float):
        """Load time series data from file(s)"""
        if isinstance(data_path, dict):
            # Separate train/test files
            path = data_path[self.split]
            data_np = np.load(path)
        else:
            # Single file - load and split
            full_data_np = np.load(data_path)
            num_train = int(len(full_data_np) * train_ratio)
            
            # Reproducible split
            rng = np.random.default_rng(self.seed)
            indices = np.arange(len(full_data_np))
            rng.shuffle(indices)
            
            if self.split == "train":
                data_np = full_data_np[indices[:num_train]]
            else:
                data_np = full_data_np[indices[num_train:]]
        
        # Convert to tensor (num_samples, 2, seq_len)
        self.data = torch.tensor(data_np, dtype=torch.float32).permute(0, 2, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return index, self.data[index]

    @staticmethod
    def _generate_trajectory_dataset(
        num_trajectories: int = 10000,
        system_type: str = "stable_spiral",
        num_timepoints: int = 100,
        t_span: Tuple[float, float] = (0, 10),
        x0_range: Tuple[float, float] = (-2, 2),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate trajectory dataset for a specific system type"""
        # System matrix definitions
        system_matrices = {
            "stable_node": np.diag([-1.2, -0.8]),
            "unstable_node": np.diag([1.5, 0.9]),
            "saddle": np.diag([-1.0, 0.8]),
            "stable_spiral": np.array([[-0.2, -2.0], [2.0, -0.2]]),
            "unstable_spiral": np.array([[0.2, -2.0], [2.0, 0.2]]),
            "center": np.array([[0, -1.5], [1.5, 0]])
        }
        
        A = system_matrices[system_type]
        time = np.linspace(t_span[0], t_span[1], num_timepoints)
        
        if seed is not None:
            np.random.seed(seed)
        
        trajectories = []
        system_dynamics = lambda t, z: A @ z  # Linear dynamics
        
        for _ in range(num_trajectories):
            x0 = np.random.uniform(*x0_range, size=2)
            sol = solve_ivp(system_dynamics, t_span, x0, t_eval=time)
            trajectories.append(sol.y.T)  # Shape: (seq_len, 2)
        
        return np.array(trajectories)  # Shape: (num_trajectories, seq_len, 2)
