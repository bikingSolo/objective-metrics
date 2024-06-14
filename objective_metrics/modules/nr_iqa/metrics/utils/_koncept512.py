import os
import time
import logging
import gdown
from typing import Tuple
from pathlib import Path

def download_checkpoints(metric: str, logger: logging.Logger) -> Tuple[str, str]:
    """
    Returns
    -------
    Tuple[str, str]
        paths to checkpoints.
    """
    ckpts_dir = Path.home() / ".cache" / "objective_metrics" / "checkpoints"
    koncept_ckpts_dir = ckpts_dir / "KonCept512"

    inception_ckpt_pth = koncept_ckpts_dir / "inceptionresnetv2-520b38e4.pth"
    koncept_ckpt_pth = koncept_ckpts_dir / "KonCept512.pth"

    os.makedirs(ckpts_dir, exist_ok=True)

    if not (os.path.isfile(inception_ckpt_pth) and os.path.isfile(koncept_ckpt_pth)):
        start = time.time()
        if logger is not None:
            logger.info(f"Downloading checkpoints for {metric} to {koncept_ckpts_dir} ...")
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1N6pRJVBOWv2UuASiAxmgt1tFADa6W56U?usp=drive_link",
            output=str(koncept_ckpts_dir)
        )
        if logger is not None:
            logger.info(f"Done in {time.time() - start} seconds")
    elif logger is not None:
        logger.info(f"Initializing {metric} with checkpoints from {koncept_ckpts_dir}.")

    return str(inception_ckpt_pth), str(koncept_ckpt_pth)