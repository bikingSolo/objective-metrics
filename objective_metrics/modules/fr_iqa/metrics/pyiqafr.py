"""
PyIQA module.

src: https://github.com/chaofengc/IQA-PyTorch
"""
import numpy as np
import pyiqa
import torch
from PIL import Image
from typing import Union, Literal
from torchvision.transforms.functional import to_tensor, resize

class PyIqa:
    def __init__(
            self,
            metric: Literal[
                "psnr", "ssim", "ms_ssim", "topiq_fr", "ahiq", "pieapp",
                "lpips", "dists", "ckdn", "fsim", "ssim", "ms_ssim", "cw_ssim",
                "psnr", "vif", "gmsd", "nlpd", "vsi", "mad"]
    ) -> None:
        """Init module with metric."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = metric
        if metric in ["psnr", "ssim", "ms_ssim", "cw_ssim"]:
            self.model = pyiqa.create_metric(metric, test_y_channel=True, color_space='ycbcr', device=self.device, as_loss=False)
        else:
            self.model = pyiqa.create_metric(metric, device=self.device, as_loss=False)

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
        if self.metric == "ckdn":
            tmp = img_ref
            img_ref = img_dist
            img_dist = tmp
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

