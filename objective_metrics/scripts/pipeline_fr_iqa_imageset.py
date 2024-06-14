"""Compute Full Reference IQA metrics for Image Dataset. Reslts in .csv format."""
# import torch
import gc
import os
import csv
import time
import logging
import numpy as np
from PIL import Image
from typing import Literal, Dict, List

from ..modules.fr_iqa.metrics.piqfr import PIQ
from ..modules.fr_iqa.metrics.pyiqafr import PyIqa

class PipelineFrIqaImageset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            img_pairs: Dict[str, List[str]],
            output: str,
            metrics: List[Literal[
                "topiq_fr", "ahiq", "pieapp", "lpips", "dists", "ckdn", "fsim",
                "ssim", "ms_ssim", "cw_ssim", "psnr", "vif", "gmsd", "nlpd", "vsi",
                "mad", "srsim", "dss", "haarpsi", "mdsi", "msgmsd"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of computing several FR IQA metrics for image dataset.

        Parameters
        ----------
        dataset_pth : str
            Path to dataset directory
        dataset_name : str
            dataset name
        img_pairs : Dict[str, List[str]], optional
            relative for dataset_pth image pairs paths
            {'ref_img_pth_1': ['dist_img_pth_1', ...,'dist_img_pth_n'], ..., 'ref_img_pth_n': [...]}
        output : str
            path to output directory
        metrics : 
            list of metrics to compute
        logger : logging.Logger
            logger
        """
        self.dataset_pth = dataset_pth
        self.dataset_name = dataset_name
        self.img_pairs = img_pairs
        self.output = output
        self.metrics = metrics
        self.logger = logger

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
        for ref_img_pth, list_dist_img_pth in self.img_pairs.items():
            self.logger.info(f"Computing {[model.metric for model in models]} for {self.dataset_name} ref: {ref_img_pth} - {ref_cnt}/{len(self.img_pairs)}")

            ref_img_fp = os.path.join(self.dataset_pth, ref_img_pth)

            ref_img_reading_error = False
            if not os.path.isfile(ref_img_fp):
                self.logger.error(f"Can't compute {[model.metric for model in models]} for {self.dataset_name} ref: {ref_img_pth} - {ref_cnt}/{len(self.img_pairs)}. File doesn't exist!")
                ref_img_reading_error = True

            if not ref_img_reading_error:
                self.logger.info(f"Reading ref image...")
                try:
                    ref_image = Image.open(ref_img_fp).convert("RGB")
                except:
                    self.logger.error(f"Can't compute {[model.metric for model in models]} for {self.dataset_name} ref: {ref_img_pth} - {ref_cnt}/{len(self.img_pairs)}. File exists but can't read reference!")
                    ref_img_reading_error = True

            num_dist_imgs = len(list_dist_img_pth)
            ref_cnt += 1
            cnt = 1

            for dist_img_pth in list_dist_img_pth:

                if not ref_img_reading_error:
                    self.logger.info(f"Distorted image: {cnt}/{num_dist_imgs}: {dist_img_pth}. ref: {ref_cnt}/{len(self.img_pairs)} - {ref_img_pth}")

                    dist_img_fp = os.path.join(self.dataset_pth, dist_img_pth)

                    dist_img_reading_error = False
                    if not os.path.isfile(dist_img_fp):
                        self.logger.error(f"Can't read distorted image: {dist_img_pth}. File doesn't exist!")
                        dist_img_reading_error = True

                    if not dist_img_reading_error:
                        self.logger.info(f"Reading dist image...")
                        try:
                            dist_image = Image.open(dist_img_fp).convert("RGB")
                        except:
                            self.logger.exception(f"File exists but can't read distorted image: {dist_img_pth}")
                            dist_img_reading_error = True

                for model in models:
                    score = np.nan
                    start = time.time()

                    if not ref_img_reading_error and not dist_img_reading_error:
                        self.logger.info(f"Computing {model.metric}.")
                        try:
                            score = model.score_img(ref_image, dist_image)
                            if score is None or np.isnan(score):
                                raise Exception("score is NA")
                        except:
                            self.logger.exception(f"Can't compute {model.metric}. Ref: {ref_img_pth}, Dist: {dist_img_pth}")

                        gc.collect()
                        # torch.cuda.empty_cache()

                    with open(
                        os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                        mode="a", newline=""
                    ) as fh:
                        csv_writer = csv.writer(fh)
                        csv_writer.writerow([dist_img_pth, score])

                    self.logger.info(f"Done in {time.time() - start:.2f} seconds\n")
                cnt += 1

    def run(self) -> None:
        """
            Init metrics and run pipeline.
            Note that all metric models will be in memory (may be gpu) !
        """
        self.logger.info(f"FR IQA Metric list: {self.metrics} for {self.dataset_name}")
        self.logger.info(f"START Initializing models")
        all_start = time.time()
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
