"""
1. For each datasets make datasets_configs/dataset.yaml
2. Run this script to compute specified metrics for each dataset from cli args

                                      |-> nr_vqa
                                      |
                                      |-> fr_vqa
               |-> vid, metric_type --|
               |                      |-> fr_iqa
               |                      |
dataset_type --|                      |-> nr_iqa
               |
               |                      |-> fr (iqa)
               |-> img, metric_type --|
                                      |-> nr (iqa)
"""

import os
import yaml
import json
import logging
import argparse
from .scripts.pipeline_fr_iqa_imageset import PipelineFrIqaImageset
from .scripts.pipeline_fr_iqa_videoset import PipelineFrIqaVideoset
from .scripts.pipeline_nr_iqa_imageset import PipelineNrIqaImageset
from .scripts.pipeline_nr_iqa_videoset import PipelineNrIqaVideoset
from .scripts.pipeline_nr_vqa_videoset import PipelineNrVqaVideoset
from .scripts.pipeline_fr_vqa_videoset import PipelineFrVqaVideoset
from .scripts.pipeline_image_features_imageset import PipelineFeaturesImageset
from .scripts.pipeline_image_features_videoset import PipelineImageFeaturesVideoset

DEBUG_VIDEO = bool(os.environ.get("DEBUG_VIDEO", False))

def run():
    ## Read Dataset Names
    parser = argparse.ArgumentParser(
        prog='objective_metrics_run.py', 
        description='Runs metrics for datasets in pipeline',
        epilog='''For usage make a dir ./datasets_config with dataset_name.yaml config file for each dataset.
        In config you should specify list of metrics and additional info (look docs).
        '''
    )
    parser.add_argument("datasets", action="extend", nargs="+")
    args = parser.parse_args()

    ## Make Logger
    fileh_root = logging.FileHandler(f'{os.getpid()}_logs.log', 'a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileh_root.setFormatter(formatter)
    fileh_root.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)

    log = logging.getLogger()
    log.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)
    log.addHandler(fileh_root)

    log.info(f"STAGE 0: Trying to compute metrics for {args.datasets}")

    ## For Each Dataset
    for dataset in args.datasets:

        log.info(f"STAGE 1: Reading {dataset} config...")

        ## Read Config
        try:
            with open(os.path.join("datasets_configs", dataset + ".yaml"), "r") as f:
                dataset_conf = yaml.safe_load(f)
        except:
            log.exception("Bad Config!")
            continue

        log.info(f"STAGE 2: Parsing {dataset} config...")

        ## Parse Config
        try:
            dataset_type = dataset_conf.pop("type")
            dataset_pth = dataset_conf.pop("dataset_pth")
            dataset_name = dataset_conf.pop("dataset_name")
            output = dataset_conf.pop("output")
            logs_output = dataset_conf.pop("logs_output")
        except:
            log.exception("Bad Config!")
            continue
        
        log.info(f"STAGE 3: Computing metrics for {dataset}...")

        ## Type of Dataset
        match dataset_type:
            case "img":
                
                # <-------- FR Metrics -------->
                if dataset_conf.get("fr_metrics") is not None:
                    metrics = dataset_conf.pop("fr_metrics")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_pairs.json"), "r") as f:
                            img_pairs = json.load(f)
                    except:
                        log.exception("No Image Pairs for FR IQA metrics!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)
                    
                    # Pipeline
                    PipelineFrIqaImageset(dataset_pth, dataset_name, img_pairs, output, metrics, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)

                # <-------- NR Metrics -------->
                if dataset_conf.get("nr_metrics") is not None:
                    metrics = dataset_conf.pop("nr_metrics")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_list.json"), "r") as f:
                            img_list = json.load(f)
                    except:
                        log.exception("No Image List for NR IQA metrics!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)

                    # Pipeline
                    PipelineNrIqaImageset(dataset_pth, dataset_name, img_list, output, metrics, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)

                # <-------- Feature Extractors -------->
                if dataset_conf.get("feature_extractors") is not None:
                    models = dataset_conf.pop("feature_extractors")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_list.json"), "r") as f:
                            img_list = json.load(f)
                    except:
                        log.exception("No Image List for feature extractors!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)

                    # Pipeline
                    PipelineFeaturesImageset(dataset_pth, dataset_name, img_list, output, models, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)

            case "vid":
                
                ## <-------- FR IQA Metrics -------->
                if dataset_conf.get("fr_iqa_metrics") is not None:
                    metrics = dataset_conf.pop("fr_iqa_metrics")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_pairs.json"), "r") as f:
                            video_pairs = json.load(f)
                    except:
                        log.exception("No Video Pairs for FR IQA metrics!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)
                    
                    # Pipeline
                    PipelineFrIqaVideoset(dataset_pth, dataset_name, video_pairs, output, metrics, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)

                ## <-------- NR IQA Metrics -------->
                if dataset_conf.get("nr_iqa_metrics") is not None:
                    metrics = dataset_conf.pop("nr_iqa_metrics")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_list.json"), "r") as f:
                            vid_list = json.load(f)
                    except:
                        log.exception("No Video List for NR IQA metrics!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)

                    # Pipeline
                    PipelineNrIqaVideoset(dataset_pth, dataset_name, vid_list, output, metrics, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)

                ## <-------- FR VQA Metrics -------->
                if dataset_conf.get("fr_vqa_metrics") is not None:
                    metrics = dataset_conf.pop("fr_vqa_metrics")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_pairs.json"), "r") as f:
                            video_pairs = json.load(f)
                    except:
                        log.exception("No Video Pairs for FR VQA metrics!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)
                    
                    # Pipeline
                    PipelineFrVqaVideoset(dataset_pth, dataset_name, video_pairs, output, metrics, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)
                    

                ## <-------- NR VQA Metrics -------->
                if dataset_conf.get("nr_vqa_metrics") is not None:
                    metrics = dataset_conf.pop("nr_vqa_metrics")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_list.json"), "r") as f:
                            vid_list = json.load(f)
                    except:
                        log.exception("No Video List for NR VQA metrics!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)

                    # Pipeline
                    PipelineNrVqaVideoset(dataset_pth, dataset_name, vid_list, output, metrics, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)


                ## <-------- Image Feature Extractors -------->
                if dataset_conf.get("image_feature_extractors") is not None:
                    models = dataset_conf.pop("image_feature_extractors")
                    try:
                        with open(os.path.join("datasets_configs", f"{dataset}_list.json"), "r") as f:
                            vid_list = json.load(f)
                    except:
                        log.exception("No Video List for image feature extractors!")
                        continue

                    # Change Logger Handler
                    fileh = logging.FileHandler(logs_output, 'a')
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fileh.setFormatter(formatter)
                    fileh.setLevel(logging.DEBUG if DEBUG_VIDEO else logging.INFO)
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh)

                    # Pipeline
                    PipelineImageFeaturesVideoset(dataset_pth, dataset_name, vid_list, output, models, log).run()

                    # Return old Handler
                    for hdlr in log.handlers[:]:
                        log.removeHandler(hdlr)
                    log.addHandler(fileh_root)

        log.info(f"DONE for {dataset}.")
