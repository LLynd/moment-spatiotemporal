from pathlib import Path
from tqdm import tqdm
from wandb.apis.public import File
from typing import Iterable, List, Tuple, Any, Dict
import omegaconf
import numpy as np
from utils.misc import from_0255_to_01
from PIL import Image
import torch
from lightning import Fabric
import torch.distributed as dist

import re
import threading
import wandb
import logging

log = logging.getLogger(__name__)

# default value in torchvision.utils.make_grid used by wandb
PAD_PIXELS = 2

MAX_THREADS = 5
semaphore = threading.Semaphore(MAX_THREADS)


def setup_wandb(config):
    """
    Sets up W&B run based on config.
    """
    from experiment.train_graph_econder_decoder_reconstruction import fabric as fabric_global

    if fabric_global.global_rank != 0:
        return

    group, name = config.exp.log_dir.parts[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        dir=config.exp.log_dir,
        group=group,
        name=name,
        config=wandb_config,
        sync_tensorboard=True,
    )


def wandb_log_sequential(log_dict: dict):
    from experiment.optimize import fabric as fabric_global
    import pickle

    # log only from rank 0
    if fabric_global.global_rank == 0:
        wandb.log(log_dict)

    log_dict = fabric_global.all_gather(log_dict)

    if fabric_global.global_rank == 0:
        wandb.log(log_dict)

    fabric_global.barrier()


def download_file_task(file: File, output_dir: Path, semaphore: threading.Semaphore):
    with semaphore:
        file.download(output_dir, exist_ok=True)


def download_files(files: Iterable[File], output_dir: Path):
    threads = []

    for file in files:
        thread = threading.Thread(
            target=download_file_task, args=(file, output_dir, semaphore)
        )
        threads.append(thread)
        thread.start()

    for thread in tqdm(threads, desc="Downloading files"):
        thread.join()

    files = [output_dir / file.name for file in files]
    return files


def select_filetype(files: Iterable[File], filetype_regex: str):
    output_files = []
    keys = {}
    for file in files:
        match = re.match(filetype_regex, file.name)
        if match is not None:
            output_files.append(file)
            keys[file.name] = int(match.group(1))

    output_files.sort(key=lambda x: keys[x.name])
    return output_files


def apply_filter(files: Iterable[Path], filter: Iterable[bool]):
    return [file for file, keep in zip(files, filter) if keep]


def split_grid(images_grid: np.array, image_size: int) -> np.array:
    n_rows = int((images_grid.shape[0] - PAD_PIXELS) / (image_size + PAD_PIXELS))
    n_cols = int((images_grid.shape[1] - PAD_PIXELS) / (image_size + PAD_PIXELS))

    if n_rows <= 1 and n_cols <= 1:
        return images_grid[None, ...]
    else:
        """
        remove last padding

        ppppp
        pipip
        ppppp

        becomes

        pppp
        pipi
        """

        images_grid = images_grid[
            : n_rows * (image_size + PAD_PIXELS),
            : n_cols * (image_size + PAD_PIXELS),
            :,
        ]

    images = []

    for i in range(n_rows):
        for j in range(n_cols):
            image = images_grid[
                i * (image_size + PAD_PIXELS) + PAD_PIXELS : (i + 1)
                * (image_size + PAD_PIXELS),
                j * (image_size + PAD_PIXELS) + PAD_PIXELS : (j + 1)
                * (image_size + PAD_PIXELS),
            ]
            images.append(image)

    return np.array(images)


def download_run(
    entity: str,
    project: str,
    run_id: str,
    save_dir: str,
    fabric: Fabric,
    metrics: List[str] = [],
) -> Tuple[str, List[Tuple[np.array, np.array]], Any, int, Dict[str, np.array]]:
    """
    Downloads images from W&B run and returns them as a list of tuples with original image and flattened grid of generated images.

    Args:
        entity: W&B entity name
        project: W&B project name
        run_id: W&B run id
        save_dir: Directory to save run files
        fabric: Fabric instance to place images on proper device
        metrics: list of metrics to download from run
    """
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)

    # beat gans
    if "image_size" in run.config["denoising_network"]:
        image_size = int(run.config["denoising_network"]["image_size"])
    # lightning-dit
    else:
        image_size = int(
            run.config["autoencoder"]["config"]["model"]["params"]["ddconfig"][
                "resolution"
            ]
        )

    tmp_dir = Path(save_dir) / project / run_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    images_original_dir = tmp_dir / "images_original"
    images_generated_dir = tmp_dir / "images_generated"

    images_original_dir.mkdir(parents=True, exist_ok=True)
    images_generated_dir.mkdir(parents=True, exist_ok=True)

    log.info("Accessing files")
    files = list(tqdm(run.files(), desc="Accessing files"))

    images_original_files_to_download = select_filetype(
        files, r"^media/images/images/original_(\d+)"
    )
    images_generated_files_to_download = select_filetype(
        files, r"^media/images/images/generated_(\d+)"
    )

    images_original_files = download_files(
        images_original_files_to_download, images_original_dir
    )
    images_generated_files = download_files(
        images_generated_files_to_download, images_generated_dir
    )

    images_original = [
        from_0255_to_01(torch.from_numpy(np.array(Image.open(file))).to(fabric.device))
        for file in images_original_files
    ]
    images_generated = [
        from_0255_to_01(
            torch.from_numpy(split_grid(np.array(Image.open(file)), image_size)).to(
                fabric.device
            )
        )
        for file in images_generated_files
    ]

    pairs = list(zip(images_original, images_generated))

    # Get all logged values for each metric
    metrics_history = {}
    for metric in metrics:
        try:
            # Get all values for this metric
            history = [
                row[metric] for row in run.scan_history(keys=[metric]) if metric in row
            ]
            metrics_history[metric] = history if history else None
        except KeyError:
            metrics_history[metric] = None
    print(metrics_history)
    metrics = metrics_history

    return tmp_dir, pairs, run.config, image_size, metrics


if __name__ == "__main__":
    sample_padded_image = np.zeros((16, 16, 1))
    sample_padded_image[2:8, 2:8, 0] = 1
    sample_padded_image[2:8, 2 + 7 : 8 + 7, 0] = 1
    sample_padded_image[2 + 7 : 8 + 7, 2:8, 0] = 1
    sample_padded_image[9:14, 9:14, 0] = 1

    splitted = split_grid(sample_padded_image, 5)

    assert splitted.shape == (4, 5, 5, 1)
    assert splitted.sum() == 5 * 5 * 4