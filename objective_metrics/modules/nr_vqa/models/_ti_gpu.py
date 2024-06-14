"""
TI gpu model.
"""
import cupy as cp
import numpy as np
from PIL import Image


def ti_gpu(frame: Image.Image, prev_frame: Image.Image) -> float:
    """Calculate TI between two frames via gpu.

    Parameters
    ----------
    frame : PIL Image
    prev_frame : PIL Image

    Returns
    -------
    float
        TI
    """
    frame = cp.asarray(np.array(frame))
    prev_frame = cp.asarray(np.array(prev_frame))

    frame_y = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    prev_frame_y = 0.299 * prev_frame[:, :, 0] + 0.587 * prev_frame[:, :, 1] + 0.114 * prev_frame[:, :, 2]

    return (frame_y - prev_frame_y).std().item()
