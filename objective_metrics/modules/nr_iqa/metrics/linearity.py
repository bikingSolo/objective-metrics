"""
Linearity module.

src: https://github.com/lidq92/LinearityIQA
"""
import numpy as np
import torch
import logging
from typing import List, Union
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_tensor

from .models._linearity import IQAModel
from .utils._linearity import download_checkpoints


class Linearity:
    def __init__(self, logger: logging.Logger = None) -> None:
        """Init Linearity module.
        """
        self.metric = "linearity"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IQAModel(
            arch="resnext101_32x8d", pool="avg", use_bn_end=False, P6=1, P7=1
        ).to(self.device)

        ckpt_pth = download_checkpoints(self.metric, logger)

        checkpoint = torch.load(ckpt_pth, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.k = checkpoint["k"]
        self.b = checkpoint["b"]

    def score_img(self, img: Union[Image.Image, str]) -> float:
        """Score one image.

        Parameters
        ----------
        img : Union[Image.Image, str]
            PIL image or path to image

        Returns
        -------
        float
            score
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        img = resize(img, (498, 664))

        with torch.no_grad():
            img = to_tensor(img).to(self.device)
            img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            score = self.model(img.unsqueeze(0))

        score = score[-1].item() * self.k[-1] + self.b[-1]

        return score

    def score_img_batch(self, img_list: Union[List[Image.Image], List[str]]) -> List[float]:
        """Score batch of images.

        Parameters
        ----------
        img_list : Union[List[Image.Image], List[str]]
            List of PIL images or paths to images.

        Returns
        -------
        List[float]
            scores
        """
        batch_list = []
        for img in img_list:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            img = resize(img, (498, 664))
            img = to_tensor(img).to(self.device)
            img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            batch_list.append(img)

        batch = torch.stack(batch_list, dim=0).to(self.device)

        with torch.no_grad():
            y = self.model(batch)

        res = [
            y[-1].tolist()[i][0] * self.k[-1] + self.b[-1]
            for i in range(len(batch_list))
        ]

        return res
