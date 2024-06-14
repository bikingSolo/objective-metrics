"""Compute Full Reference VQA metrics for Video Dataset. Reslts in .csv format."""
import os
import csv
import time
import logging
import subprocess
import numpy as np
from typing import Literal, Dict, List, Tuple

from ..modules.fr_vqa.metrics.vmaf import VMAF

class PipelineFrVqaVideoset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            video_pairs: Dict[str, List[Tuple[List[str], Literal["same", "repeat_each", "repeat_last"]]]],
            output: str,
            metrics: List[Literal["vmaf"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of computing several FR VQA metrics for video dataset.

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

        os.makedirs(output, exist_ok=True)
        
    def run_metric(self, model: VMAF) -> None:
        """Compute metric for dataset. Save results in .csv

        Parameters
        ----------
        model : VMAF
            initialized metric model
        """
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

            self.logger.info(f"Computing {model.metric} for {self.dataset_name} Ref: {ref_vid_pth} - {ref_cnt}/{len(self.video_pairs)}")
            self.logger.info(f"Number of distorted video sets: {len(dist_vid_params_list)}")

            all_start = time.time()
            for set_num, dist_vid_params in enumerate(dist_vid_params_list):

                dist_vid_pths, mode = dist_vid_params
                dist_vid_fullpaths = [os.path.join(self.dataset_pth, dist_vid_pth) for dist_vid_pth in dist_vid_pths]

                self.logger.info(f"Computing distorted videos in set number {set_num + 1}/{len(dist_vid_params_list)}")
                self.logger.info(f"Number of distorted videos in set number {set_num + 1}/{len(dist_vid_params_list)} : {len(dist_vid_pths)}")
                
                for i, dist_vid_fullpath in enumerate(dist_vid_fullpaths):

                    self.logger.info(f"Computing {i + 1}/{len(dist_vid_pths)} distorted video in set number {set_num + 1}/{len(dist_vid_params_list)}")
                    self.logger.info(f"Ref: {ref_vid_pth}, Dist: {dist_vid_pths[i]}, Mode: {mode}")

                    start = time.time()

                    score = np.nan
                    try:
                        score = model.score_video(ref_vid_fullpath, dist_vid_fullpath, mode)
                    except subprocess.CalledProcessError as e:
                        self.logger.exception(f"Can't compute {model.metric}. Ref: {ref_vid_pth}, Dist: {dist_vid_pths[i]}")
                        self.logger.error(f"Shell command error output: {e.output}")
                    except:
                        self.logger.exception(f"Can't compute {model.metric}. Ref: {ref_vid_pth}, Dist: {dist_vid_pths[i]}")

                    with open(
                        os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                        mode="a", newline=""
                    ) as fh:
                        csv_writer = csv.writer(fh)
                        csv_writer.writerow([os.path.relpath(dist_vid_fullpath, self.dataset_pth), score])

                    self.logger.info(f"Done in {time.time() - start:.2f} seconds\n")

            self.logger.info(f"Computed all distorted videos for Ref: {ref_vid_pth} in {time.time() - all_start:.2f} seconds\n")

    def run(self) -> None:
        """Run Pipeline.
        """
        self.logger.info(f"FR VQA Metric list: {self.metrics} for {self.dataset_name}")
        all_start = time.time()
        for i, metric in enumerate(self.metrics):
            try:
                if metric == "vmaf":
                    model = VMAF(self.logger)
                else:
                    raise Exception(f"Unknown model {metric}!")

            except:
                self.logger.exception(f"Can't init {metric}. Pass on")
                continue

            self.logger.info(f"START Computing {metric}. Number {i + 1} metric out of {len(self.metrics)}")
            start = time.time()
            self.run_metric(model)
            self.logger.info(f"DONE Computing {metric} in {time.time() - start:.2f} seconds.")
        self.logger.info(f"ALL FR VQA Metrics DONE for {self.dataset_name} in {time.time() - all_start:.2f} seconds.")
