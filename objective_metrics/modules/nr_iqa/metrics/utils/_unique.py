import os
import gdown
import time
import logging
from pathlib import Path

def download_checkpoints(metric: str, logger: logging.Logger) -> str:
    """
    Returns
    -------
    str
        path to checkpoint.
    """

    ckpts_dir = Path.home() / ".cache" / "objective_metrics" / "checkpoints"
    unique_ckpts_dir = ckpts_dir / "UNIQUE"
    unique_ckpt_pth = unique_ckpts_dir / "model.pt"

    os.makedirs(unique_ckpts_dir, exist_ok=True)

    if not os.path.isfile(unique_ckpt_pth):
        start = time.time()
        if logger is not None:
            logger.info(f"Downloading checkpoints for {metric} to {unique_ckpts_dir} ...")
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1d_FYZb8XRBeq_WNqf8bKxuOCNgLUirfN?usp=drive_link", 
            output=str(unique_ckpts_dir)
        )
        if logger is not None:
            logger.info(f"Done in {time.time() - start} seconds")
    elif logger is not None:
        logger.info(f"Initializing {metric} with checkpoints from {unique_ckpts_dir}.")

    return str(unique_ckpt_pth)