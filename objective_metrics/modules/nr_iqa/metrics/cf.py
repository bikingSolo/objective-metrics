"""
CF module.
"""
import torch
from PIL import Image
from typing import Union


class CF:
    def __init__(self) -> None:
        """Init CF module.
        """
        if torch.cuda.is_available():
            try:
                from .models._cf_gpu import CFGpuModel
                self.model = CFGpuModel()
            except:
                from .models._cf import CFModel
                self.model = CFModel()
        else:
            from .models._cf import CFModel
            self.model = CFModel()
        self.metric = "cf"

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
        return self.model.score_img(img)
