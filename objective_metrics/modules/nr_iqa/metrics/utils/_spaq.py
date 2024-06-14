"""
Utility for SPAQ.
"""
import os
import logging
import time
import gdown
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms


def download_checkpoints(metric: str, logger: logging.Logger) -> str:
    """
    Returns
    -------
    str
        path to checkpoint.
    """
    spaq_map = {"spaq-bl": "BL_release.pt", "spaq-mta": "MT-A_release.pt", "spaq-mts": "MT-S_release.pt"}
    url_map = {
        "spaq-bl": "https://drive.google.com/file/d/13ZHXKtL9RaGTlpm6qPlBZ_S1YxBOJ0gw/view?usp=drive_link",
        "spaq-mta": "https://drive.google.com/file/d/16tXAF-oi9YddYHV2ea3kIWMSe_g65X45/view?usp=drive_link",
        "spaq-mts": "https://drive.google.com/file/d/1IgTxDEl3VoAaKz7GhIJHLpySfy0axm56/view?usp=drive_link" 
    }
    ckpts_dir = Path.home() / ".cache" / "objective_metrics" / "checkpoints"
    spaq_ckpts_dir = ckpts_dir / "SPAQ"
    spaq_ckpt_pth = spaq_ckpts_dir / spaq_map[metric]
    url = url_map[metric]

    os.makedirs(spaq_ckpts_dir, exist_ok=True)

    if not os.path.isfile(spaq_ckpt_pth):
        start = time.time()
        if logger is not None:
            logger.info(f"Downloading checkpoints for {metric} to {spaq_ckpts_dir} ...")
        gdown.download(
            url, fuzzy=True,
            output=str(spaq_ckpt_pth)
        )
        if logger is not None:
            logger.info(f"Done in {time.time() - start} seconds")
    elif logger is not None:
        logger.info(f"Initializing {metric} with checkpoints from {spaq_ckpts_dir}.")

    return str(spaq_ckpt_pth)


class Image_load(object):
    def __init__(self, size, stride, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.stride = stride
        self.interpolation = interpolation

    def __call__(self, img):
        image = self.adaptive_resize(img)
        return self.generate_patches(image, input_size=self.stride)

    def adaptive_resize(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            Torch Tensor: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            return transforms.ToTensor()(img)
        else:
            return transforms.ToTensor()(
                transforms.Resize(self.size, self.interpolation)(img)
            )

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))

    def generate_patches(self, image, input_size, type=np.float32):
        img = self.to_numpy(image)
        img_shape = img.shape
        img = img.astype(dtype=type)
        if len(img_shape) == 2:
            (
                H,
                W,
            ) = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray(
                [
                    img,
                ]
                * 3,
                dtype=img.dtype,
            )

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [
            img[hId : hId + input_size, wId : wId + input_size, :]
            for hId in hIdx
            for wId in wIdx
        ]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)
