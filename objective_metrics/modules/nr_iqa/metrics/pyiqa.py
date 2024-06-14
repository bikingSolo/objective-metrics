"""
PyIQA module.

src: https://github.com/chaofengc/IQA-PyTorch
"""
import pyiqa
import torch
from typing import Literal, Union, List
from PIL import Image
from torchvision.transforms.functional import to_tensor


class PyIqa:
    def __init__(
            self,
            metric: Literal[
                "niqe", "dbcnn", "brisque", "ilniqe", "maniqa",
                "musiq", "nrqm", "pi", "nima", "clipiqa", "clipiqa+",
                "cnniqa", "tres", "paq2piq", "hyperiqa", "topiq_nr"]
    ) -> None:
        """Init PyIQA module.

        Parameters
        ----------
        metric : 
            What metric to init.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = metric
        self.model = pyiqa.create_metric(metric, device=device, as_loss=False)

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

        with torch.no_grad():
            img = to_tensor(img)
            score = self.model(img.unsqueeze(0))
        
        score = score.item()

        return score

    def score_img_batch(self, img_list: Union[List[Image.Image], List[str]]) -> List[float]:
        """Score batch of images.

        Parameters
        ----------
        img_list : Union[List[Image.Image], List[str]]
            list of paths to images or PIL images

        Returns
        -------
        List[float]
            scores
        """
        batch_list = []
        for img in img_list:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            img = to_tensor(img)
            batch_list.append(img)

        batch = torch.stack(batch_list, dim=0)

        with torch.no_grad():
            y = self.model(batch)

        res = y.tolist()

        return res
