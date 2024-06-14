"""
SI gpu model.
"""
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import sobel
from PIL import Image
from typing import Union


class SIGpuModel:
    def __init__(self) -> None:
        """Init SIGpuModel.
        """
        pass

    def score_img(self, img: Union[Image.Image, str]) -> float:
        """Score one image.

        Parameters
        ----------
        img : Union[Image.Image, str]
            path to image or PIL image

        Returns
        -------
        float
            score
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        img = cp.asarray(np.array(img))

        Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        sobel_h = sobel(Y, axis=0)
        sobel_v = sobel(Y, axis=1)

        # crop output to valid window, calculate gradient magnitude
        magnitude = cp.sqrt(sobel_h**2 + sobel_v**2)[1:-1, 1:-1]
        
        return magnitude.std().item()
