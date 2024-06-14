# import torch
import gc
import logging
import numpy as np
import cv2
import time
from typing import Tuple, Union, List
from PIL import Image

from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu

from .metrics.koncept512 import KonCept512
from .metrics.linearity import Linearity
from .metrics.pyiqa import PyIqa
from .metrics.spaq import Spaq
from .metrics.unique import UNIQUE
from .metrics.si import SI
from .metrics.cf import CF


# Deprecated due to high memory consumption!
def run_one_video_raw_yuv420(
        pth: str, 
        model: Union[KonCept512, Linearity, PyIqa, Spaq, UNIQUE, SI, CF], 
        size: Tuple[int, int],
        logger: logging.Logger
) -> float:
    """Score video in raw yuv420 format.

    Parameters
    ----------
    pth : str
        path to video
    model : Union[KonCept512, Linearity, PyIqa, Spaq, UNIQUE, SI, CF]
        initialized metric model
    size : Tuple[int, int]
        spatial resolution of video
    logger : logging.Logger

    Returns
    -------
    float
        score
    """
    score = 0
    frame_count = 0
    with open(pth, "rb") as video:
        while True:
            img = video.read(int(1.5 * size[0] * size[1]))
            if not img:
                break

            frame = np.frombuffer(img, dtype=np.uint8).reshape(-1, size[1])
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)

            try:
                score += model.score_img(Image.fromarray(frame))
                frame_count += 1
            except:
                logger.warning(f"Error on {frame_count + 1} frame. Vid: {pth}", exc_info=True)

    return score / frame_count


def run_one_video(
    pth: str,
    models: List[Union[KonCept512, Linearity, PyIqa, Spaq, UNIQUE, SI, CF]],
    read_video: VideoReaderDecord | VideoReaderOpenCVcpu,
    logger: logging.Logger
) -> List[float]:
    """Score video via averaging scores over frames.

    Parameters
    ----------
    pth : str
        path to video
    models : List[Union[KonCept512, Linearity, PyIqa, Spaq, UNIQUE, SI, CF]]
        initialized metric models
    read_video : VideoReaderDecord | VideoReaderOpenCVcpu
        generator over video frames
    logger : logging.Logger

    Returns
    -------
    List[float]
        list of scores of each model in the same order as models
    """
    logger.info("START initializing video reader...")
    start = time.time()

    reader = read_video(pth)

    logger.info(f"DONE initializing video reader in {time.time() - start} seconds.")
    logger.info(f"Number of frames: {len(reader)}")

    scores = [0 for _ in models]
    lucky_frames_counts = [0 for _ in models]
    for frame_num, frame in enumerate(reader):
        logger.debug(f"START for frame {frame_num} ...")
        frame_start = time.time()
        for i, model in enumerate(models):
            try:
                score = model.score_img(frame)
                if score is None or np.isnan(score):
                    raise Exception("score is NA")
                scores[i] += score
                lucky_frames_counts[i] += 1
            except:
                logger.warning(f"Error for {model.metric} on {frame_num} frame. Video: {pth}", exc_info=True)
            gc.collect()
            # torch.cuda.empty_cache()
        logger.debug(f"DONE for frame {frame_num} in {time.time() - frame_start} seconds.")

    for i, lucky_frames_count in enumerate(lucky_frames_counts):
        if lucky_frames_count == 0:
            logger.warning(f"Error for {models[i].metric} - no frames computed. Score = nan. Video: {pth}")

    return [score / lucky_frames_count if lucky_frames_count != 0 else np.nan for score, lucky_frames_count in zip(scores, lucky_frames_counts)]
