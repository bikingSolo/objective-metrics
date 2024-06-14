# import torch
import logging
import os
import gc
import time
import numpy as np
from typing import Literal, Union, List, Dict
from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu

from .metrics.piqfr import PIQ
from .metrics.pyiqafr import PyIqa


def space_evenly(length: int, n_dup: int) -> None | np.ndarray:
    """Space n frames evenly over [0, length - 1] range.

    Parameters
    ----------
    length : int
        length of a range
    n_dup : int
        number of frames

    Returns
    -------
    None | np.ndarray
        array with frame indexes to repeat
    """
    if n_dup == 0:
        return None
    n = int(length / n_dup)
    return np.linspace(0, n * (n_dup - 1), n_dup, dtype=int)

def get_repeat_numbers(ref_len: int, dist_len: int) -> np.ndarray:
    """Get repeat number for each distorted video frame.

    Parameters
    ----------
    ref_len : int
        number of frames in reference video
    dist_len : int
        number of frames in distorted video

    Returns
    -------
    np.ndarray
        Array with dist_len length. Each element is a number of repetitions.
    """
    integer_number_repeats = ref_len // dist_len
    remain_frames_number = ref_len - integer_number_repeats * dist_len

    repeats = np.array([integer_number_repeats] * dist_len)
    extra_repeats = space_evenly(dist_len, remain_frames_number)

    if extra_repeats is not None:
        repeats[extra_repeats] += 1

    return repeats

def run_videos(
        ref_pth: str,
        dist_pths: List[str],
        models: List[Union[PIQ, PyIqa]],
        mode: Literal["same", "repeat_each", "repeat_last"],
        video_reader: VideoReaderDecord | VideoReaderOpenCVcpu,
        logger: logging.Logger
    ) -> Dict[str, List[float]]:
    """Compute FR metrics for videos frame to frame and averaging.

    Algorithm will compute frame to frame scores untill both videos (or only one of them) ends.
    If one of source or distorted videos is shorter, then warning will be thrown in log file.

    To avoid this in situation when distorted videos are shorter (may be because of frame dropping),
    you can use one of below options: 

    if mode == "same":
        No frames will be repeated.

    If mode == "repeat_each":
        If distorted videos are n times shorter than reference,
        than each frame of all distorted videos will be repeated n times.
        If n isn't an integer then frames to repeat will be evenly spaced.

    If mode == "repeat_last":
        In distorted videos last frame will be repeated as many times as needed to match ref.

    If source video is shorter, then use mode == "same" to simply drop excess frames of distorted videos.
    That's may be suitable, for example, when distorted videos have stalls.

    Parameters
    ----------
    ref_pth : str
        path to reference video
    dist_pths : List[str]
        paths to distorted videos with the same length
    models : Union[PIQ, PyIqa]
        initialized metric models
    mode : Literal["same", "repeat_each", "repeat_last"]
        how to process when distorted and reference videos has different lengths
    video_reader : VideoReaderDecord | VideoReaderOpenCVcpu,
        generator over video frames
    logger : logging.Logger, optional
    Returns
    -------
    Dict[str, List[float]]
        score for each model in the same order as models for each distorted video
    """
    if mode not in ["same", "repeat_each", "repeat_last"]:
        logger.error(f"Unknown mode={mode}. Provide one of: same, repeat_each, repeat_last")
        return {dist_pth: [np.nan for _ in models] for dist_pth in dist_pths}

    logger.info("START initializing video readers...")
    start = time.time()

    if not os.path.isfile(ref_pth):
        logger.error(f"Ref video: {ref_pth} doesn't exist!")
        return {dist_pth: [np.nan for _ in models] for dist_pth in dist_pths}
    ref_reader = video_reader(ref_pth)
    
    dist_readers, scores, lucky_frames_counts = {}, {}, {}
    for dist_pth in dist_pths:
        scores[dist_pth] = [0 for _ in models]
        lucky_frames_counts[dist_pth] = [0 for _ in models]
        if os.path.isfile(dist_pth):
            dist_readers[dist_pth] = video_reader(dist_pth)
        else:
            logger.warning(f"Dist video: {dist_pth} doesn't exist!")

    if len(dist_readers) == 0:
        logger.error(f"All dist videos can't be read!")
        return {dist_pth: [np.nan for _ in models] for dist_pth in dist_pths}
    
    logger.info(f"DONE initializing video readers in {time.time() - start} seconds.")

    ref_len = len(ref_reader)
    logger.info(f"Reference number of frames: {ref_len}")
    dist_len_min = min([len(dist_readers[dist_pth]) for dist_pth in dist_pths])
    dist_len_max = max([len(dist_readers[dist_pth]) for dist_pth in dist_pths])
    logger.info(f"Distorted min number of frames: {dist_len_min}")
    logger.info(f"Distorted max number of frames: {dist_len_max}")

    logger.info(f"Computing {len(dist_pths)} dist videos for ref: {ref_pth}. Mode: {mode}")

    if mode == "repeat_each":
        logger.info("Сalculating number of repetitions for each frame...")
        repeat_numbers = get_repeat_numbers(ref_len, dist_len_min)
        logger.info("Start computing...")

    ref_end = False # To exit when source video is shorter
    repeat_number = 1
    i, ref_frame_number = 0, 0
    for dist_frames in zip(*dist_readers.values()):
        if ref_end:
            break
        dist_frames = dict(zip(dist_readers.keys(), dist_frames))
        if mode == "repeat_each":
            repeat_number = repeat_numbers[i]
        i += 1
        for _ in range(repeat_number):
            logger.debug(f"START for ref frame {ref_frame_number} ...")
            frame_start = time.time()
            ref_frame = next(ref_reader, None)
            if ref_frame is None:
                logger.warning(f"Source video is shorter than some of distorted!", exc_info=True)
                ref_end = True
                break
            ref_frame_number += 1
            for dist_pth, dist_frame in dist_frames.items():
                for model_num, model in enumerate(models):
                    try:
                        score = model.score_img(ref_frame, dist_frame)
                        if score is None or np.isnan(score):
                            raise Exception("score is NA")
                        scores[dist_pth][model_num] += score
                        lucky_frames_counts[dist_pth][model_num] += 1
                    except:
                        logger.warning(f"Error for {model.metric} on {ref_frame_number - 1} frame. Ref: {ref_pth}, Dist: {dist_pth}", exc_info=True)
                    gc.collect()
                    # torch.cuda.empty_cache()
            logger.debug(f"DONE for ref frame {ref_frame_number - 1} in {time.time() - frame_start} seconds.")

    if mode == "repeat_last":
        for ref_frame in ref_reader:
            logger.debug(f"START for ref frame {ref_frame_number} ...")
            frame_start = time.time()
            ref_frame_number += 1
            for dist_pth, dist_frame in dist_frames.items():
                for model_num, model in enumerate(models):
                    try:
                        score = model.score_img(ref_frame, dist_frame)
                        if score is None or np.isnan(score):
                            raise Exception("score is NA")
                        scores[dist_pth][model_num] += score
                        lucky_frames_counts[dist_pth][model_num] += 1
                    except:
                        logger.warning(f"Error for {model.metric} on {ref_frame_number - 1} frame. Ref: {ref_pth}, Dist: {dist_pth}", exc_info=True)
                    gc.collect()
                    # torch.cuda.empty_cache()
            logger.debug(f"DONE for ref frame {ref_frame_number - 1} in {time.time() - frame_start} seconds.")

    if next(ref_reader, None) is not None: 
        logger.warning("source video isn't finished!")
    for dist_pth, dist_reader in dist_readers.items():
        if (next(dist_reader, None) is not None):
            logger.warning(f"dist video isn't finished: {dist_pth}!")

    for dist_pth in dist_pths:
        for i, model in enumerate(models):
            if lucky_frames_counts[dist_pth][i] == 0:
                logger.warning(f"Error for {model.metric} - no frames computed. Score = nan. Ref: {ref_pth}, Dist: {dist_pth}")

    
    return {
        dist_pth: [scores[dist_pth][i] / lucky_frames_counts[dist_pth][i] if lucky_frames_counts[dist_pth][i] != 0 else np.nan for i in range(len(models))] 
        for dist_pth in dist_pths
    }
