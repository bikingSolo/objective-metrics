"""
CF cpu model.
"""
import numpy as np
from PIL import Image
from typing import Union


class CFModel:
    def __init__(self) -> None:
        """Init CFModel.
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
        img = np.array(img)

        rg = img[:, :, 0] - img[:, :, 1]
        by = 0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2]
        std_rgby = np.sqrt(rg.std()**2 + by.std()**2)
        mean_rgby = np.sqrt(rg.mean()**2 + by.mean()**2)

        return std_rgby + 0.3 * mean_rgby
