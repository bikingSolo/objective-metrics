"""
UNIQUE module.

src: https://github.com/zwx8981/UNIQUE/tree/master
"""
import numpy as np
import torch
import logging
from torchvision import transforms
from typing import Union
from PIL import Image

from .models._unique import BaseCNN, AdaptiveResize, parse_config
from .utils._unique import download_checkpoints


test_transform = transforms.Compose([
    AdaptiveResize(768),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

class UNIQUE:
    def __init__(self, logger: logging.Logger = None) -> None:
        """Init UNIQUE module.

        Parameters
        ----------
        ckpt_pth : str
            path to UNIQUE checkpoint.
        """
        self.metric = "unique"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = parse_config()
        config.backbone = 'resnet34'
        config.representation = 'BCNN'

        self.model = BaseCNN(config)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        ckpt_pth = download_checkpoints(self.metric, logger)

        checkpoint = torch.load(ckpt_pth, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.model.eval().to(self.device)

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

        img = test_transform(img)
        img = torch.unsqueeze(img, dim=0)
    
        with torch.no_grad():
            img = img.to(self.device)
            score, _ = self.model(img)

        score = score.item()

        return score
