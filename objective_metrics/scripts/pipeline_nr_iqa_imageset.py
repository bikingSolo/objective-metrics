"""Compute No Reference IQA metrics for Image Dataset. Reslts in .csv format."""
# import torch
import gc
import os
import csv
import time
import logging
import numpy as np
from PIL import Image
from typing import Literal, List

from ..modules.nr_iqa.metrics.koncept512 import KonCept512
from ..modules.nr_iqa.metrics.linearity import Linearity
from ..modules.nr_iqa.metrics.pyiqa import PyIqa
from ..modules.nr_iqa.metrics.spaq import Spaq
from ..modules.nr_iqa.metrics.unique import UNIQUE
from ..modules.nr_iqa.metrics.si import SI
from ..modules.nr_iqa.metrics.cf import CF

class PipelineNrIqaImageset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            img_list: List[str],
            output: str,
            metrics: List[Literal[
                "niqe", "koncept512", "linearity", "paq2piq", "musiq", "spaq-bl",
                "spaq-mta", "spaq-mts", "dbcnn", "brisque", "ilniqe", "maniqa",
                "nrqm", "pi", "clipiqa", "clipiqa+", "cnniqa", "tres", "hyperiqa",
                "nima", "unique", "topiq_nr", "si", "cf"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of computing several NR IQA metrics for image dataset.

        Parameters
        ----------
        dataset_pth : str
            Path to dataset directory
        dataset_name : str
            dataset name
        img_list : List[str]
            relative for dataset_pth image paths
            ['img_pth_1', ...,'img_pth_n']
        output : str
            path to output directory
        metrics : 
            list of metrics to compute
        logger : logging.Logger
            logger
        """
        self.dataset_pth = dataset_pth
        self.dataset_name = dataset_name
        self.img_list = img_list
        self.output = output
        self.metrics = metrics
        self.logger = logger

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

        num_dist_imgs = len(self.img_list)
        for cnt, dist_img_pth in enumerate(self.img_list):
            self.logger.info(f"Computing {[model.metric for model in models]} for {self.dataset_name} dist: {dist_img_pth} - {cnt + 1}/{num_dist_imgs}")

            dist_img_fp = os.path.join(self.dataset_pth, dist_img_pth)

            reading_error = False
            if not os.path.isfile(dist_img_fp):
                self.logger.error(f"Can't compute {[model.metric for model in models]}. Dist: {dist_img_fp}. File doesn't exist!")
                reading_error = True

            if not reading_error:
                self.logger.info(f"Reading image ...")
                try:
                    image = Image.open(dist_img_fp).convert("RGB")
                except:
                    self.logger.exception(f"File exists but can't read image: {dist_img_pth}")
                    reading_error = True

            for model in models:
                score = np.nan
                start = time.time()

                if not reading_error:
                    self.logger.info(f"Computing {model.metric} for {self.dataset_name} dist: {dist_img_pth} - {cnt + 1}/{num_dist_imgs}")
                    try:
                        score = model.score_img(image)
                        if score is None or np.isnan(score):
                            raise Exception("score is NA")
                    except:
                        self.logger.exception(f"Can't compute {model.metric}. Dist: {dist_img_pth}")

                    gc.collect()
                    # torch.cuda.empty_cache()

                with open(
                    os.path.join(self.output, f"{self.dataset_name}_{model.metric}.csv"),
                    mode="a", newline=""
                ) as fh:
                    csv_writer = csv.writer(fh)
                    csv_writer.writerow([dist_img_pth, score])

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
