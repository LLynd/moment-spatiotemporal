import torch
from typing import Callable, Optional
#from lightning import Fabric
import wandb
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

'''
@torch.no_grad()
def calculate_distance_matrix_batched(
    inputs: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fabric: Fabric,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Vectorized batch processing - fastest for GPU-based metrics like LPIPS
    """
    n_imgs = inputs.shape[0]
    distances = torch.zeros(n_imgs, n_imgs, device=fabric.device)

    # Preprocess inputs if needed (convert H,W,C to C,H,W)
    if len(inputs.shape) == 4 and inputs.shape[-1] in [
        1,
        3,
    ]:  # Assuming last dim is channels
        inputs = inputs.permute(0, 3, 1, 2)

    # Get upper triangular indices
    idx_r, idx_c = torch.triu_indices(n_imgs, n_imgs, offset=1, device=fabric.device)
    n_pairs = len(idx_r)

    # Process in batches
    for start_idx in range(0, n_pairs, batch_size):
        end_idx = min(start_idx + batch_size, n_pairs)
        batch_idx_r = idx_r[start_idx:end_idx]
        batch_idx_c = idx_c[start_idx:end_idx]

        # Batch select inputs
        left_batch = inputs[batch_idx_r]
        right_batch = inputs[batch_idx_c]

        # Calculate distances for the batch
        batch_distances = []
        for left, right in zip(left_batch, right_batch):
            dist = metric(left, right).flatten()
            batch_distances.append(dist)

        batch_distances = torch.stack(batch_distances)

        # Assign to distance matrix
        distances[batch_idx_r, batch_idx_c] = batch_distances.flatten()
        distances[batch_idx_c, batch_idx_r] = batch_distances.flatten()  # Symmetric

    from utils.wandb import wandb_log_sequential

    wandb_log_sequential({"distance_matrix": wandb.Image(distances.cpu().numpy())})
    return distances


@torch.no_grad()
def calculate_distance_matrix_vectorized(
    inputs: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fabric: Fabric,
) -> torch.Tensor:
    """
    Fully vectorized approach - works well for simple metrics like L2
    """
    n_imgs = inputs.shape[0]

    # Preprocess inputs if needed
    if len(inputs.shape) == 4 and inputs.shape[-1] in [1, 3]:
        inputs = inputs.permute(0, 3, 1, 2)

    # For L2 distance, we can compute all pairs at once
    if hasattr(metric, "__name__") and "l2" in metric.__name__.lower():
        # Reshape for broadcasting: (n, 1, *) - (1, n, *) = (n, n, *)
        inputs_expanded = inputs.unsqueeze(1)  # (n, 1, C, H, W)
        inputs_transposed = inputs.unsqueeze(0)  # (1, n, C, H, W)

        # Compute pairwise differences
        diff = inputs_expanded - inputs_transposed  # (n, n, C, H, W)
        distances = torch.norm(diff.view(n_imgs, n_imgs, -1), dim=2)  # (n, n)

        # Zero out diagonal
        distances.fill_diagonal_(0)

    else:
        # Fallback to batched approach for complex metrics
        return calculate_distance_matrix_batched(inputs, metric, fabric)

    from utils.wandb import wandb_log_sequential

    wandb_log_sequential({"distance_matrix": wandb.Image(distances.cpu().numpy())})
    return distances


def compute_distance_chunk(args):
    """Helper function for multiprocessing"""
    inputs, indices, metric = args
    distances = []

    for idx_r, idx_c in tqdm(indices, desc="Computing distances"):
        left, right = inputs[idx_r], inputs[idx_c]
        if len(left.shape) == 3:
            left, right = left.permute(2, 0, 1), right.permute(2, 0, 1)

        dist = metric(left, right).flatten()
        distances.append((idx_r.item(), idx_c.item(), dist.item()))

    return distances


@torch.no_grad()
def calculate_distance_matrix_multiprocess(
    inputs: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fabric: Fabric,
    n_workers: int = None,
) -> torch.Tensor:
    """
    Multiprocessing approach - good for CPU-intensive metrics
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    n_imgs = inputs.shape[0]
    distances = torch.zeros(n_imgs, n_imgs, device=fabric.device)

    # Move inputs to CPU for multiprocessing
    inputs_cpu = inputs.cpu()

    # Get upper triangular indices
    idx_r, idx_c = torch.triu_indices(n_imgs, n_imgs, offset=1)
    indices = list(zip(idx_r, idx_c))

    # Split work among processes
    chunk_size = max(1, len(indices) // n_workers)
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    # Prepare arguments for each process
    args = [(inputs_cpu, chunk, metric) for chunk in chunks]

    # Process in parallel
    with mp.Pool(n_workers) as pool:
        results = pool.map(compute_distance_chunk, args)

    # Collect results
    for chunk_results in results:
        for idx_r, idx_c, dist in chunk_results:
            distances[idx_r, idx_c] = dist
            distances[idx_c, idx_r] = dist  # Symmetric

    from utils.wandb import wandb_log_sequential

    wandb_log_sequential({"distance_matrix": wandb.Image(distances.cpu().numpy())})
    return distances


@torch.no_grad()
def calculate_distance_matrix_threaded(
    inputs: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fabric: Fabric,
    n_threads: int = 4,
) -> torch.Tensor:
    """
    Threading approach - good balance for I/O bound operations
    """
    n_imgs = inputs.shape[0]
    distances = torch.zeros(n_imgs, n_imgs, device=fabric.device)

    # Preprocess inputs if needed
    if len(inputs.shape) == 4 and inputs.shape[-1] in [1, 3]:
        inputs = inputs.permute(0, 3, 1, 2)

    idx_r, idx_c = torch.triu_indices(n_imgs, n_imgs, offset=1, device=fabric.device)

    def compute_distance(idx):
        r_idx, c_idx = idx_r[idx], idx_c[idx]
        left, right = inputs[r_idx], inputs[c_idx]
        return idx, metric(left, right).flatten()

    # Use ThreadPoolExecutor for parallel computation
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(compute_distance, i) for i in range(len(idx_r))]

        for future in tqdm(futures, desc="Computing distances"):
            idx, dist = future.result()
            r_idx, c_idx = idx_r[idx], idx_c[idx]
            distances[r_idx, c_idx] = dist
            distances[c_idx, r_idx] = dist  # Symmetric

    from utils.wandb import wandb_log_sequential

    wandb_log_sequential({"distance_matrix": wandb.Image(distances.cpu().numpy())})
    return distances


# Convenience function to choose the best approach
@torch.no_grad()
def calculate_distance_matrix_auto(
    inputs: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fabric: Fabric,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Automatically choose the best parallelization strategy
    """
    n_imgs = inputs.shape[0]

    # Check if metric is likely vectorizable (simple L2)
    if hasattr(metric, "__name__") and any(
        name in metric.__name__.lower() for name in ["l2", "mse", "euclidean"]
    ):
        return calculate_distance_matrix_vectorized(inputs, metric, fabric)

    # For GPU-based metrics like LPIPS, use batched approach
    if fabric.device.type == "cuda":
        return calculate_distance_matrix_batched(inputs, metric, fabric, batch_size)

    # For CPU, use threading for moderate sizes, multiprocessing for large
    if n_imgs < 200:
        return calculate_distance_matrix_threaded(inputs, metric, fabric)
    else:
        return calculate_distance_matrix_multiprocess(inputs, metric, fabric)

'''
def from_m1p1_to_01(x):
    is_m1 = (x.flatten(start_dim=1).min(dim=1)[0] < 0.0).any()
    if is_m1:
        x = x - x.flatten(start_dim=1).min(dim=1)[0].view(-1, 1, 1, 1)
        x = x / x.flatten(start_dim=1).max(dim=1)[0].view(-1, 1, 1, 1)
    else:
        log.warning("Input is not in [-1, 1] range!")
    return x


def from_01_to_m1p1(x):
    is_0 = (x.flatten(start_dim=1).min(dim=1)[0] >= 0.0).any()
    if is_0:
        x = (x - 0.5) * 2
    else:
        log.warning("Input is not in [0, 1] range!")
    return x


def from_0255_to_01(x):
    is_255 = (x.flatten(start_dim=1).max(dim=1)[0] > 1.0).any()
    if is_255:
        x = x / 255.0
    else:
        log.warning("Input is not in [0, 255] range!")
    return x


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def should_log(t: Optional[torch.Tensor], log_frequency: int) -> bool:
    # if t is None it is infinity optimization
    if t is None:
        return True

    t_value = t.unique().item()  # Assuming t is a tensor with a single unique value

    # Check if value is effectively an integer
    if (
        isinstance(t_value, int) or abs(t_value - round(t_value)) < 1e-6
    ):  # Handle float precision issues
        # Integer case in range [0, T]
        return int(round(t_value)) % log_frequency == 0
    else:
        # Float case in range [0, 1)
        # For continuous case, log at evenly spaced intervals
        # If log_frequency is e.g. 10, log at 0.0, 0.1, 0.2, etc.
        log_points = 1.0 / log_frequency
        return any(
            abs(t_value - (i * log_points)) < 0.01 * log_points
            for i in range(log_frequency)
        )


def min_max_scale(tensor):
    B = tensor.shape[0]
    tensor = tensor - tensor.flatten(start_dim=1).min(1)[0].view(B, 1, 1, 1)
    tensor = tensor / tensor.flatten(start_dim=1).max(1)[0].view(B, 1, 1, 1)
    return tensor