"""Compute No Reference IQA metrics for Video Dataset. Reslts in .csv format."""
import os
import csv
import time
import decord
import logging
import numpy as np
from functools import partial
from typing import Literal, List
from ..modules.nr_iqa.video import run_one_video
from ..modules.utils.video_readers import VideoReaderOpenCVcpu, VideoReaderDecord

from ..modules.nr_iqa.metrics.koncept512 import KonCept512
from ..modules.nr_iqa.metrics.linearity import Linearity
from ..modules.nr_iqa.metrics.pyiqa import PyIqa
from ..modules.nr_iqa.metrics.spaq import Spaq
from ..modules.nr_iqa.metrics.unique import UNIQUE
from ..modules.nr_iqa.metrics.si import SI
from ..modules.nr_iqa.metrics.cf import CF

class PipelineNrIqaVideoset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            vid_list: List[str],
            output: str,
            metrics: List[Literal[
                "niqe", "koncept512", "linearity", "paq2piq", "musiq", "spaq-bl",
                "spaq-mta", "spaq-mts", "dbcnn", "brisque", "ilniqe", "maniqa",
                "nrqm", "pi", "clipiqa", "clipiqa+", "cnniqa", "tres", "hyperiqa",
                "nima", "unique", "topiq_nr", "si", "cf"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of computing several NR IQA metrics for video dataset.

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
        
    def run_metrics(self, models: List[KonCept512 | Linearity | PyIqa | Spaq | UNIQUE | SI | CF]) -> None:
        """Compute metrics for dataset. Save results in .csv

        Parameters
        ----------
        models : List[KonCept512 | Linearity | PyIqa | Spaq | UNIQUE | SI | CF]
            initialized metric models
        """
        for model in models:
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

            self.logger.info(f"Computing {[model.metric for model in models]} for {self.dataset_name} dist: {dist_vid_pth} - {cnt}/{num_dist_vids}")

            start = time.time()
            scores = [np.nan for _ in models]
            decord_error = False
            file_exist = True

            if not os.path.isfile(dist_vid_fp):
                self.logger.error(f"Can't compute {[model.metric for model in models]}. Dist: {dist_vid_pth}. File doesn't exist!")
                file_exist = False

            if self.decord_gpu_build and not self.use_cv2 and file_exist:
                try:
                    self.logger.info(f"Trying with Decord GPU video reader...")
                    scores = run_one_video(dist_vid_fp, models, partial(VideoReaderDecord, mode="gpu"), self.logger)
                except (decord.DECORDError, decord.DECORDLimitReachedError):
                    self.logger.exception(f"Can't compute {[model.metric for model in models]} for distorted video: {dist_vid_pth} with Decord GPU reader!")
                    decord_error = True
                except:
                    self.logger.exception(f"Can't compute {[model.metric for model in models]} for distorted video: {dist_vid_pth} !")

            if (decord_error or not self.decord_gpu_build) and not self.use_cv2 and file_exist:
                self.logger.info(f"Trying with Decord CPU video reader...")
                try:
                    scores = run_one_video(dist_vid_fp, models, partial(VideoReaderDecord, mode="cpu"), self.logger)
                    decord_error = False
                except (decord.DECORDError, decord.DECORDLimitReachedError):
                    self.logger.exception(f"Can't compute {[model.metric for model in models]} for distorted video: {dist_vid_pth} with Decord CPU reader!")
                    decord_error = True
                except:
                    self.logger.exception(f"Can't compute {[model.metric for model in models]} for distorted video: {dist_vid_pth} !")
                    decord_error = False
            
            if (decord_error or self.use_cv2) and file_exist:
                self.logger.info(f"Trying with OpenCV CPU video reader...")
                try:
                    scores = run_one_video(dist_vid_fp, models, VideoReaderOpenCVcpu, self.logger)
                except:
                    self.logger.exception(f"Can't compute {[model.metric for model in models]} for distorted video: {dist_vid_pth} with OpenCV CPU reader!")
                    
            for i, model in enumerate(models):
                with open(
                    os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                    mode="a", newline=""
                ) as fh:
                    csv_writer = csv.writer(fh)
                    csv_writer.writerow([dist_vid_pth, scores[i]])

            self.logger.info(f"Done in {time.time() - start:.2f} seconds\n")

    def run(self) -> None:
        """
            Init metrics and run pipeline.
            Note that all metric models will be in memory (may be gpu) !
        """
        self.logger.info(f"NR IQA Metric list: {self.metrics} for {self.dataset_name}")
        self.logger.info(f"START Initializing models")
        all_start = time.time()
        models = []
        for metric in self.metrics:
            self.logger.info(f"Initializing model for {metric} ...")
            start = time.time()
            try:
                if metric in ["niqe", "dbcnn", "brisque", "ilniqe", "maniqa", "musiq", "nrqm", "pi", "nima", 
                            "clipiqa", "clipiqa+", "cnniqa", "tres", "paq2piq","hyperiqa", "topiq_nr"]:
                    model = PyIqa(metric)

                elif metric == "koncept512":
                    model = KonCept512(self.logger)

                elif metric == "linearity":
                    model = Linearity(self.logger)

                elif metric in ["spaq-bl", "spaq-mta", "spaq-mts"]:
                    model = Spaq(metric, self.logger)

                elif metric == "unique":
                    model = UNIQUE(self.logger)

                elif metric == "si":
                    model = SI()

                elif metric == "cf":
                    model = CF()

                else:
                    raise Exception(f"Unknown model {metric}!")
            except:
                self.logger.exception(f"Can't init {metric}. Pass on")
                continue
            
            self.logger.info(f"Done for {metric} in {time.time() - start:.2f} seconds.")
            models.append(model)

        self.logger.info(f"DONE initializing in {time.time() - all_start:.2f} seconds.")
        
        if len(models) == 0:
            self.logger.warning("No NR IQA metrics are succesfully initialized!")
        else:
            start = time.time()
            self.run_metrics(models)
            self.logger.info(f"ALL NR IQA Metrics DONE for {self.dataset_name} in {time.time() - start:.2f} seconds.")
