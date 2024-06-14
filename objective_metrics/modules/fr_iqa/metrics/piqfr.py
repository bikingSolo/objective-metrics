"""
PIQ module.

src: https://github.com/photosynthesis-team/piq?tab=readme-ov-file
"""
import numpy as np
import torch
import piq
from PIL import Image
from typing import Union, Literal
from torchvision.transforms.functional import to_tensor, resize

class PIQ:
    def __init__(self, metric: Literal["srsim", "dss", "haarpsi", "mdsi", "msgmsd"]) -> None:
        """Init module with metric."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = metric
        if metric == "srsim":
            self.model = piq.srsim
        elif metric == "dss":
            self.model = piq.dss
        elif metric == "haarpsi":
            self.model = piq.haarpsi
        elif metric == "mdsi":
            self.model = piq.mdsi
        elif metric == "msgmsd":
            self.model = piq.multi_scale_gmsd
        else:
            raise BaseException("This metrics is not allowed!")

    def score_img(self, img_ref: Union[str, Image.Image], img_dist: Union[str, Image.Image]) -> float:
        """
        Parameters
        ----------
        img_ref : Union[str, Image.Image]
        img_dist : Union[str, Image.Image]

        Returns
        -------
        float
            score
        """
        if isinstance(img_ref, str):
            img_ref = Image.open(img_ref).convert("RGB")
        if isinstance(img_dist, str):
            img_dist = Image.open(img_dist).convert("RGB")

        if img_dist.size != img_ref.size:
            img_dist = resize(img_dist, size=(img_ref.height, img_ref.width), interpolation=Image.BICUBIC)

        with torch.no_grad():
            img_ref = to_tensor(img_ref).to(self.device)
            img_dist = to_tensor(img_dist).to(self.device)

            score = self.model(img_ref.unsqueeze(0), img_dist.unsqueeze(0))

        score = score.item()

        return score
