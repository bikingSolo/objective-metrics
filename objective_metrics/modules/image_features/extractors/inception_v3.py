"""
Inception v3 image feature extractor class.
"""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_tensor, center_crop

from .models._inception_v3 import InceptionHeadless


class InceptionExtractor:
    def __init__(self) -> None:
        """Init InceptionExtractor model.
        """
        self.name = "inception_v3"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionHeadless().to(self.device)
        self.model.eval()

    def get_img_features(self, img: Image.Image | str) -> np.ndarray:
        """Get features of one image.

        Parameters
        ----------
        img : Image.Image | str
            PIL Image or path to image.

        Returns
        -------
        np.ndarray
            features vector
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        img = resize(img, [342])
        img = center_crop(img, [299])

        with torch.no_grad():
            img = to_tensor(img).to(self.device)
            img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            features = self.model(img.unsqueeze(0))[0].to("cpu").numpy()

        return features
