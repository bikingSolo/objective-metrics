"""Extract deep features from each image of Image Dataset. Reslts in .npz format."""
import gc
import os
import time
import logging
import numpy as np
from PIL import Image
from typing import Literal, List

from ..modules.image_features.extractors.inception_v3 import InceptionExtractor

class PipelineFeaturesImageset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            img_list: List[str],
            output: str,
            models: List[Literal["inception_v3"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of extracting features with several models for image dataset.

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
        models : 
            list of models to use
        logger : logging.Logger
            logger
        """
        self.dataset_pth = dataset_pth
        self.dataset_name = dataset_name
        self.img_list = img_list
        self.output = output
        self.models = models
        self.logger = logger

        os.makedirs(output, exist_ok=True)
        
    def run_models(self, models: List[InceptionExtractor]) -> None:
        """Compute metrics for dataset. Save results in .npz

        Parameters
        ----------
        models : List[InceptionExtractor]
            initialized models
        """
        models_results_dict = {}
        for model in models:
            models_results_dict[f"{self.dataset_name}_{model.name}-features.npz"] = {}

        num_dist_imgs = len(self.img_list)
        for cnt, dist_img_pth in enumerate(self.img_list):
            self.logger.info(f"Computing {[model.name for model in models]} for {self.dataset_name} dist: {dist_img_pth} - {cnt + 1}/{num_dist_imgs}")

            dist_img_fp = os.path.join(self.dataset_pth, dist_img_pth)

            reading_error = False
            if not os.path.isfile(dist_img_fp):
                self.logger.error(f"Can't compute {[model.name for model in models]}. Dist: {dist_img_fp}. File doesn't exist!")
                reading_error = True

            if not reading_error:
                self.logger.info(f"Reading image ...")
                try:
                    image = Image.open(dist_img_fp).convert("RGB")
                except:
                    self.logger.exception(f"File exists but can't read image: {dist_img_pth}")
                    reading_error = True

            for model in models:
                feature_vec = np.nan
                start = time.time()

                if not reading_error:
                    self.logger.info(f"Computing {model.name} for {self.dataset_name} dist: {dist_img_pth} - {cnt + 1}/{num_dist_imgs}")
                    try:
                        feature_vec = model.get_img_features(image)
                        if feature_vec is None or np.isnan(feature_vec).any():
                            raise Exception("feature vector is NA")
                    except:
                        self.logger.exception(f"Can't compute {model.name}. Dist: {dist_img_pth}")

                    gc.collect()
                    # torch.cuda.empty_cache()

                models_results_dict[f"{self.dataset_name}_{model.name}-features.npz"][dist_img_pth] = feature_vec

                self.logger.info(f"Done in {time.time() - start:.2f} seconds\n")

        for filename, results in models_results_dict.items():
            np.savez(os.path.join(self.output, filename), **results)

    def run(self) -> None:
        """
            Init models and run pipeline.
            Note that all models will be in memory (may be gpu)!
        """
        self.logger.info(f"Image feature extraction models list: {self.models} for {self.dataset_name}")
        self.logger.info(f"START Initializing models")
        all_start = time.time()
        models = []
        for model_name in self.models:
            self.logger.info(f"Initializing model for {model_name} ...")
            start = time.time()
            try:
                if model_name == "inception_v3":
                    model = InceptionExtractor()

                else:
                    raise Exception(f"Unknown model {model_name}!")
            except:
                self.logger.exception(f"Can't init {model_name}. Pass on")
                continue

            self.logger.info(f"Done for {model_name} in {time.time() - start:.2f} seconds.")
            models.append(model)
        
        self.logger.info(f"DONE initializing in {time.time() - all_start:.2f} seconds.")

        if len(models) == 0:
            self.logger.warning("No feature extraction models are succesfully initialized!")
        else:
            start = time.time()
            self.run_models(models)
            self.logger.info(f"ALL feature extraction models DONE for {self.dataset_name} in {time.time() - start:.2f} seconds.")
