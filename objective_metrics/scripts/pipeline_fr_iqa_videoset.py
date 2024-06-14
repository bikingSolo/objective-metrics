"""Compute Full Reference IQA metrics for Video Dataset. Reslts in .csv format."""
import os
import csv
import time
import decord
import logging
import numpy as np
from functools import partial
from typing import Literal, Dict, List, Tuple

from ..modules.utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu
from ..modules.fr_iqa.video import run_videos
from ..modules.fr_iqa.metrics.piqfr import PIQ
from ..modules.fr_iqa.metrics.pyiqafr import PyIqa

class PipelineFrIqaVideoset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            video_pairs: Dict[str, List[Tuple[List[str], Literal["same", "repeat_each", "repeat_last"]]]],
            output: str,
            metrics: List[Literal[
                "topiq_fr", "ahiq", "pieapp", "lpips", "dists", "ckdn", "fsim",
                "ssim", "ms_ssim", "cw_ssim", "psnr", "vif", "gmsd", "nlpd", "vsi",
                "mad", "srsim", "dss", "haarpsi", "mdsi", "msgmsd"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of computing several FR IQA metrics for video dataset.

        Parameters
        ----------
        dataset_pth : str
            Path to dataset directory
        dataset_name : str
            dataset name
        video_pairs : Dict
            {
                ref_1:
                    [
                        ([dist_1, dist_2, ...], mode_1),
                        ...,
                        ([dist_1, dist_2, ...], mode_k)
                    ], 
                ...,
                ref_k:
                    [
                        ([...], mode_1),
                        ...,
                        ([...], mode_k)
                    ]
            }
            For each reference video a list of tuples with distorted videos with the same length and mode.
            Each path should be relative to dataset_pth.
            Look at run_videos function for more.
        output : str
            path to output directory
        metrics : List
            list of metrics to compute
        logger : logging.Logger
            logger
        """
        self.dataset_pth = dataset_pth
        self.dataset_name = dataset_name
        self.video_pairs = video_pairs
        self.output = output
        self.metrics = metrics
        self.logger = logger

        self.decord_gpu_build = bool(os.environ.get("DECORD_GPU", False))
        self.use_cv2 = bool(os.environ.get("USE_CV2", False))

        os.makedirs(output, exist_ok=True)

    def run_metrics(self, models: List[PIQ | PyIqa]) -> None:
        """Compute metrics for dataset. Save results in .csv

        Parameters
        ----------
        models : List[PIQ | PyIqa]
            initialized metric models
        """
        for model in models:
            with open(
                os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                mode="w", newline=""
            ) as fh:
                csv_writer = csv.writer(fh)
                csv_writer.writerow(["filename", model.metric])

        ref_cnt = 0
        for ref_vid_pth, dist_vid_params_list in self.video_pairs.items():
            ref_vid_fullpath = os.path.join(self.dataset_pth, ref_vid_pth)
            ref_cnt += 1

            self.logger.info(f"Computing {[model.metric for model in models]} for {self.dataset_name} ref: {ref_vid_pth} - {ref_cnt}/{len(self.video_pairs)}")
            self.logger.info(f"Number of distorted video sets: {len(dist_vid_params_list)}")

            all_start = time.time()
            for set_num, dist_vid_params in enumerate(dist_vid_params_list):

                dist_vid_pths, mode = dist_vid_params
                dist_vid_fullpaths = [os.path.join(self.dataset_pth, dist_vid_pth) for dist_vid_pth in dist_vid_pths]

                self.logger.info(f"Computing distorted videos in set number {set_num + 1}/{len(dist_vid_params_list)}")
                self.logger.info(f"Number of distorted videos in set number {set_num + 1}/{len(dist_vid_params_list)} : {len(dist_vid_pths)}")
                start = time.time()
                
                scores = {dist_vid_fullpath: [np.nan for _ in models] for dist_vid_fullpath in dist_vid_fullpaths}
                decord_error = False
                
                if self.decord_gpu_build and not self.use_cv2:
                    try:
                        self.logger.info(f"Trying with Decord GPU video reader...")
                        scores = run_videos(ref_vid_fullpath, dist_vid_fullpaths, models, mode, partial(VideoReaderDecord, mode="gpu"), self.logger)
                    except (decord.DECORDError, decord.DECORDLimitReachedError):
                        self.logger.exception(f"Error during computing scores for {ref_vid_pth} with Decord GPU reader!")
                        decord_error = True
                    except:
                        self.logger.exception(f"Error during computing scores for {ref_vid_pth} !")
                    
                if (decord_error or not self.decord_gpu_build) and not self.use_cv2:                
                    self.logger.info(f"Trying with Decord CPU video reader...")
                    try:
                        scores = run_videos(ref_vid_fullpath, dist_vid_fullpaths, models, mode, partial(VideoReaderDecord, mode="cpu"), self.logger)
                        decord_error = False
                    except (decord.DECORDError, decord.DECORDLimitReachedError):
                        self.logger.exception(f"Error during computing scores for {ref_vid_pth} with Decord CPU reader!")
                        decord_error = True
                    except:
                        self.logger.exception(f"Error during computing scores for {ref_vid_pth} !")
                        decord_error = False

                if decord_error or self.use_cv2:
                    self.logger.info(f"Trying with OpenCV CPU video reader...")
                    try:
                        scores = run_videos(ref_vid_fullpath, dist_vid_fullpaths, models, mode, VideoReaderOpenCVcpu, self.logger)
                    except:
                        self.logger.exception(f"Error during computing scores for {ref_vid_pth} with OpenCV CPU reader!")

                for model_num, model in enumerate(models):
                    with open(
                        os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                        mode="a", newline=""
                    ) as fh:
                        csv_writer = csv.writer(fh)
                        for dist_vid_fullpath, models_scores in scores.items():
                            csv_writer.writerow([os.path.relpath(dist_vid_fullpath, self.dataset_pth), models_scores[model_num]])

                self.logger.info(f"Done in {time.time() - start:.2f} seconds\n")

            self.logger.info(f"Computed all distorted videos for ref: {ref_vid_pth} in {time.time() - all_start:.2f} seconds\n")

    def run(self) -> None:
        """
            Init metrics and run pipeline.
            Note that all metric models will be in memory (may be gpu) !
        """
        self.logger.info(f"FR IQA Metric list: {self.metrics} for {self.dataset_name}")
        self.logger.info(f"START Initializing models")
        all_start= time.time()
        models = []
        for metric in self.metrics:
            self.logger.info(f"Initializing model for {metric} ...")
            start = time.time()
            try:
                if metric in [
                    "topiq_fr", "ahiq", "pieapp", "lpips", "dists", "ckdn", "fsim", "ssim", "ms_ssim",
                    "cw_ssim", "psnr", "vif", "gmsd", "nlpd", "vsi", "mad"
                ]:
                    model = PyIqa(metric)
                elif metric in [
                    "srsim", "dss", "haarpsi", "mdsi", "msgmsd"
                ]:
                    model = PIQ(metric)
                else:
                    raise Exception(f"Unknown model {metric}!")
            except:
                self.logger.exception(f"Can't init {metric}. Pass on")
                continue

            self.logger.info(f"Done for {metric} in {time.time() - start:.2f} seconds.")
            models.append(model)
        
        self.logger.info(f"DONE initializing in {time.time() - all_start:.2f} seconds.")
    
        if len(models) == 0:
            self.logger.warning("No FR IQA metrics are succesfully initialized!")
        else:
            start = time.time()
            self.run_metrics(models)
            self.logger.info(f"ALL FR IQA Metrics DONE for {self.dataset_name} in {time.time() - start:.2f} seconds.")
