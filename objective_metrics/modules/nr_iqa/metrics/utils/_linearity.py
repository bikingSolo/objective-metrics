import os
import logging
import gdown
import time
from pathlib import Path

def download_checkpoints(metric: str, logger: logging.Logger) -> str:
    """
    Returns
    -------
    str
        path to checkpoint.
    """
    ckpts_dir = Path.home() / ".cache" / "objective_metrics" / "checkpoints"
    linearity_ckpts_dir = ckpts_dir / "Linearity"
    linearity_ckpt_pth = linearity_ckpts_dir / "p1q2plus0.1variant.pth"

    os.makedirs(ckpts_dir, exist_ok=True)

    if not os.path.isfile(linearity_ckpt_pth):
        start = time.time()
        if logger is not None:
            logger.info(f"Downloading checkpoints for {metric} to {linearity_ckpts_dir} ...")
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1cLW6eMibPl9GJWvRB4WTszc1JgfWcgJc?usp=drive_link",
            output=str(linearity_ckpts_dir)
        )
        if logger is not None:
            logger.info(f"Done in {time.time() - start} seconds")
    elif logger is not None:
        logger.info(f"Initializing {metric} with checkpoints from {linearity_ckpts_dir}.")

    return str(linearity_ckpt_pth)
