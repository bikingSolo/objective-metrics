"""Compute No Reference VQA metrics for Video Dataset. Reslts in .csv format."""
# import torch
import gc
import os
import csv
import decord
import time
import logging
import numpy as np
from functools import partial
from typing import Literal, List

from ..modules.utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu

from ..modules.nr_vqa.dover import DOVER
from ..modules.nr_vqa.fastvqa import FastVQA
from ..modules.nr_vqa.mdtvsfa import MDTVSFA
from ..modules.nr_vqa.ti import TI

class PipelineNrVqaVideoset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            vid_list: List[str],
            output: str,
            metrics: List[Literal["mdtvsfa", "FasterVQA", "FAST-VQA", "dover", "ti"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of computing several NR VQA metrics for video dataset.

        Parameters
        ----------
        dataset_pth : str
            Path to dataset directory
        dataset_name : str
            dataset name
        vid_list : List[str]
            relative for dataset_pth video paths
            ['vid_pth_1', ...,'vid_pth_n']
        output : str
            path to output directory
        metrics : 
            list of metrics to compute
        logger : logging.Logger
            logger
        """
        self.dataset_pth = dataset_pth
        self.dataset_name = dataset_name
        self.vid_list = vid_list
        self.output = output
        self.metrics = metrics
        self.logger = logger

        self.decord_gpu_build = bool(os.environ.get("DECORD_GPU", False))
        self.use_cv2 = bool(os.environ.get("USE_CV2", False))

        os.makedirs(output, exist_ok=True)
        
    def run_metric(self, model: DOVER | FastVQA | MDTVSFA | TI) -> None:
        """Compute metric for dataset. Save results in .csv

        Parameters
        ----------
        model : DOVER | FastVQA | MDTVSFA | TI
            initialized metric model
        """
        with open(
            os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
            mode="w", newline=""
        ) as fh:
            csv_writer = csv.writer(fh)
            csv_writer.writerow(["filename", model.metric])

        cnt = 0
        num_dist_vids = len(self.vid_list)
        for dist_vid_pth in self.vid_list:
            dist_vid_fp = os.path.join(self.dataset_pth, dist_vid_pth)
            cnt += 1
            self.logger.info(f"Computing {model.metric} for {self.dataset_name} dist: {dist_vid_pth} - {cnt}/{num_dist_vids}")

            start = time.time()
            score = np.nan
            decord_error = False
            file_exist = True

            if not os.path.isfile(dist_vid_fp):
                self.logger.error(f"Can't compute {model.metric}. Dist: {dist_vid_pth}. File doesn't exist!")
                file_exist = False
            
            if self.decord_gpu_build and not self.use_cv2 and file_exist:
                try:
                    self.logger.info(f"Trying with Decord GPU video reader...")
                    score = model.score_video(dist_vid_fp, partial(VideoReaderDecord, mode="gpu"))
                    if score is None or np.isnan(score):
                        raise Exception("score is NA")
                except (decord.DECORDError, decord.DECORDLimitReachedError):
                    self.logger.exception(f"Can't compute {model.metric}. Dist: {dist_vid_pth} with Decord GPU reader")
                    decord_error = True
                except:
                    self.logger.exception(f"Can't compute {model.metric}. Dist: {dist_vid_pth}")

            if (decord_error or not self.decord_gpu_build) and not self.use_cv2 and file_exist:
                self.logger.info(f"Trying with Decord CPU video reader...")
                gc.collect()
                # torch.cuda.empty_cache()
                try:
                    score = model.score_video(dist_vid_fp, partial(VideoReaderDecord, mode="cpu"))
                    if score is None or np.isnan(score):
                        raise Exception("score is NA")
                    decord_error = False
                except (decord.DECORDError, decord.DECORDLimitReachedError):
                    self.logger.exception(f"Can't compute {model.metric}. Dist: {dist_vid_pth} with Decord CPU reader")
                    decord_error = True
                except:
                    self.logger.exception(f"Can't compute {model.metric}. Dist: {dist_vid_pth}")
                    decord_error = False

            if (decord_error or self.use_cv2) and file_exist:
                self.logger.info(f"Trying with OpenCV CPU video reader...")
                gc.collect()
                # torch.cuda.empty_cache()
                try:
                    score = model.score_video(dist_vid_fp, VideoReaderOpenCVcpu)
                    if score is None or np.isnan(score):
                        raise Exception("score is NA")
                except:
                    self.logger.exception(f"Can't compute {model.metric}. Dist: {dist_vid_pth} with OpenCV CPU reader")

            gc.collect()
            # torch.cuda.empty_cache()

            with open(
                os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                mode="a", newline=""
            ) as fh:
                csv_writer = csv.writer(fh)
                csv_writer.writerow([dist_vid_pth, score])

            self.logger.info(f"Done in {time.time() - start:.2f} seconds\n")

    def run(self) -> None:
        """Run Pipeline.
        """
        self.logger.info(f"NR VQA Metric list: {self.metrics} for {self.dataset_name}")
        all_start = time.time()
        for i, metric in enumerate(self.metrics):
            try:
                if metric == "mdtvsfa":
                    model = MDTVSFA(self.logger)

                elif metric in ["FasterVQA", "FAST-VQA"]:
                    model = FastVQA(metric, self.logger)

                elif metric == "dover":
                    model = DOVER(self.logger)

                elif metric == "ti":
                    model = TI()

                else:
                    raise Exception(f"Unknown model {metric}!")
            except:
                self.logger.exception(f"Can't init {metric}. Pass on")
                continue

            self.logger.info(f"START Computing {metric}. Number {i + 1} metric out of {len(self.metrics)}")
            start = time.time()
            self.run_metric(model)
            self.logger.info(f"DONE Computing {metric} in {time.time() - start:.2f} seconds.")
        self.logger.info(f"ALL NR VQA Metrics DONE for {self.dataset_name} in {time.time() - all_start:.2f} seconds.")
