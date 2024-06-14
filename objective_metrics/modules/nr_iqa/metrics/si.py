"""
SI module.
"""
import torch
from PIL import Image
from typing import Union


class SI:
    def __init__(self) -> None:
        """Init SI module.
        """
        if torch.cuda.is_available():
            try:
                from .models._si_gpu import SIGpuModel
                self.model = SIGpuModel()
            except:
                from .models._si import SIModel
                self.model = SIModel()
        else:
            from .models._si import SIModel
            self.model = SIModel()
        self.metric = "si"

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
