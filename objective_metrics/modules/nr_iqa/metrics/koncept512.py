"""
KonCept512 module.

src: https://github.com/ZhengyuZhao/koniq-PyTorch
"""
import logging
import torch
from typing import Union, List
import numpy as np
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_tensor

from .models._koncept512 import model_qa
from .utils._koncept512 import download_checkpoints


class KonCept512:
    def __init__(self, logger: logging.Logger = None) -> None:
        """Init KonCept512 module.
        """
        self.metric = "koncept512"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_inception_pth, ckpt_koncetp_pth = download_checkpoints(self.metric, logger)

        self.KonCept512 = model_qa(num_classes=1, ckpt_pth=ckpt_inception_pth)
        self.KonCept512.load_state_dict(torch.load(ckpt_koncetp_pth, map_location=self.device))
        self.KonCept512.eval().to(self.device)

    def score_img(self, img: Union[Image.Image, str]) -> float:
        """Score one image.

        Parameters
        ----------
        img : Union[Image.Image, str]
            PIL Image or path to image.

        Returns
        -------
        float
            score
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if img.height != 384 or img.width != 512:
            img = resize(img, (384, 512))

        with torch.no_grad():
            img = to_tensor(img).to(self.device)
            img = normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            score = self.KonCept512(img.unsqueeze(0))

        score = score.item()

        return score

    def score_img_batch(self, img_list: Union[List[str], List[Image.Image]]) -> List[float]:
        """Score batch of images.

        Parameters
        ----------
        img_list : Union[List[str], List[Image.Image]]
            list of paths to images or PIL images

        Returns
        -------
        List[float]
            list of scores
        """
        batch_list = []
        for img in img_list:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if img.height != 384 or img.width != 512:
                img = resize(img, (384, 512))
            img = to_tensor(img).to(self.device)
            img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            batch_list.append(img)

        batch = torch.stack(batch_list, dim=0).to(self.device)

        with torch.no_grad():
            y = self.KonCept512(batch)

        res = y.tolist()

        return res
