"""
SPAQ module.

src: https://github.com/h4nwei/SPAQ
"""
import numpy as np
import torch
import logging
from typing import Union, Literal
from PIL import Image

from .utils._spaq import Image_load, download_checkpoints
from .models._spaq import Baseline, MTS, MTA


class Spaq:
    def __init__(self, metric: Literal["spaq-bl", "spaq-mta", "spaq-mts"], logger: logging.Logger) -> None:
        """Init Spaq Module.

        Parameters
        ----------
        metric : Literal["spaq-bl", "spaq-mta", "spaq-mts"]
            what metric to init.
        ckpt_pth : str
            path to model checkpoint.
        """
        self.metric = metric
        if metric == "spaq-bl":
            self.model = Baseline()

        elif metric == "spaq-mta":
            self.model = MTA()

        elif metric == "spaq-mts":
            self.model = MTS()


        self.prepare_image = Image_load(size=512, stride=224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        ckpt_pth = download_checkpoints(self.metric, logger)
        checkpoint = torch.load(ckpt_pth, map_location=self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

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
        img = self.prepare_image(img)
        
        with torch.no_grad():
            img = img.to(self.device)
            if self.metric == "spaq-mts":
                score, _ = self.model(img)
            else:
                score = self.model(img)
            score = score.mean().item()

        return score

    def score_img_batch(self, pth_list):
        raise Exception("No support for batch inference. Use batch size = 0")