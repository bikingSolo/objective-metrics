# import torch
import gc
import logging
import numpy as np
import time
from typing import List

from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu
from .extractors.inception_v3 import InceptionExtractor

def run_one_video(
    pth: str,
    models: List[InceptionExtractor],
    read_video: VideoReaderDecord | VideoReaderOpenCVcpu,
    logger: logging.Logger
) -> List[np.ndarray]:
    """Get video feature vectors via spatial average pooling over all frames.

    Parameters
    ----------
    pth : str
        path to video
    models : List[InceptionExtractor]
        initialized feature extractors
    read_video : VideoReaderDecord | VideoReaderOpenCVcpu
        generator over video frames
    logger : logging.Logger

    Returns
    -------
    List[np.ndarray]
        list of features for each model in the same order as models
    """
    logger.info("START initializing video reader...")
    start = time.time()

    reader = read_video(pth)

    logger.info(f"DONE initializing video reader in {time.time() - start} seconds.")
    logger.info(f"Number of frames: {len(reader)}")

    features = [None for _ in models]
    lucky_frames_counts = [0 for _ in models]
    for frame_num, frame in enumerate(reader):
        logger.debug(f"START for frame {frame_num} ...")
        frame_start = time.time()
        for i, model in enumerate(models):
            try:
                feature_vec = model.get_img_features(frame)
                if feature_vec is None or np.isnan(feature_vec).any():
                    raise Exception("feature vector is NA")
                features[i] = feature_vec if features[i] is None else features[i] + feature_vec
                lucky_frames_counts[i] += 1
            except:
                logger.warning(f"Error for {model.metric} on {frame_num} frame. Video: {pth}", exc_info=True)
            gc.collect()
            # torch.cuda.empty_cache()
        logger.debug(f"DONE for frame {frame_num} in {time.time() - frame_start} seconds.")

    for i, lucky_frames_count in enumerate(lucky_frames_counts):
        if lucky_frames_count == 0:
            logger.warning(f"Error for {models[i].metric} - no frames computed. Score = nan. Video: {pth}")

    return [feature_vec / lucky_frames_count if lucky_frames_count != 0 else np.nan for feature_vec, lucky_frames_count in zip(features, lucky_frames_counts)]
