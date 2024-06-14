"""Extract deep features from each video of Video Dataset. Reslts in .npz format."""
import os
import time
import decord
import logging
import numpy as np
from functools import partial
from typing import Literal, List
from ..modules.image_features.video import run_one_video
from ..modules.utils.video_readers import VideoReaderOpenCVcpu, VideoReaderDecord

from ..modules.image_features.extractors.inception_v3 import InceptionExtractor

class PipelineImageFeaturesVideoset:
    def __init__(
            self,
            dataset_pth: str,
            dataset_name: str,
            vid_list: List[str],
            output: str,
            models: List[Literal["inception_v3"]],
            logger: logging.Logger
    ) -> None:
        """Init pipeline of extracting features with several models for video dataset.

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
        models : 
            list of models to use
        logger : logging.Logger
            logger
        """
        self.dataset_pth = dataset_pth
        self.dataset_name = dataset_name
        self.vid_list = vid_list
        self.output = output
        self.models = models
        self.logger = logger

        self.decord_gpu_build = bool(os.environ.get("DECORD_GPU", False))
        self.use_cv2 = bool(os.environ.get("USE_CV2", False))

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

        cnt = 0
        num_dist_vids = len(self.vid_list)
        for dist_vid_pth in self.vid_list:
            dist_vid_fp = os.path.join(self.dataset_pth, dist_vid_pth)
            cnt += 1

            self.logger.info(f"Computing {[model.name for model in models]} for {self.dataset_name} dist: {dist_vid_pth} - {cnt}/{num_dist_vids}")

            start = time.time()
            feature_vectors = [np.nan for _ in models]
            decord_error = False
            file_exist = True

            if not os.path.isfile(dist_vid_fp):
                self.logger.error(f"Can't compute {[model.name for model in models]}. Dist: {dist_vid_pth}. File doesn't exist!")
                file_exist = False

            if self.decord_gpu_build and not self.use_cv2 and file_exist:
                try:
                    self.logger.info(f"Trying with Decord GPU video reader...")
                    feature_vectors = run_one_video(dist_vid_fp, models, partial(VideoReaderDecord, mode="gpu"), self.logger)
                except (decord.DECORDError, decord.DECORDLimitReachedError):
                    self.logger.exception(f"Can't compute {[model.name for model in models]} for distorted video: {dist_vid_pth} with Decord GPU reader!")
                    decord_error = True
                except:
                    self.logger.exception(f"Can't compute {[model.name for model in models]} for distorted video: {dist_vid_pth} !")

            if (decord_error or not self.decord_gpu_build) and not self.use_cv2 and file_exist:
                self.logger.info(f"Trying with Decord CPU video reader...")
                try:
                    feature_vectors = run_one_video(dist_vid_fp, models, partial(VideoReaderDecord, mode="cpu"), self.logger)
                    decord_error = False
                except (decord.DECORDError, decord.DECORDLimitReachedError):
                    self.logger.exception(f"Can't compute {[model.name for model in models]} for distorted video: {dist_vid_pth} with Decord CPU reader!")
                    decord_error = True
                except:
                    self.logger.exception(f"Can't compute {[model.name for model in models]} for distorted video: {dist_vid_pth} !")
                    decord_error = False
            
            if (decord_error or self.use_cv2) and file_exist:
                self.logger.info(f"Trying with OpenCV CPU video reader...")
                try:
                    feature_vectors = run_one_video(dist_vid_fp, models, VideoReaderOpenCVcpu, self.logger)
                except:
                    self.logger.exception(f"Can't compute {[model.name for model in models]} for distorted video: {dist_vid_pth} with OpenCV CPU reader!")
                    

            for i, model in enumerate(models):
                models_results_dict[f"{self.dataset_name}_{model.name}-features.npz"][dist_vid_pth] = feature_vectors[i]

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
