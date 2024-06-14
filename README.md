[![PyPI](https://img.shields.io/pypi/v/objective-metrics)](https://pypi.org/project/objective-metrics/)

# Introduction

It is a tool for convenient use of objective quality metrics via the command line. You can use it to run calculations on whole datasets via GPU or CPU and track the results.

There is support of No Reference (NR) and Full Reference (FR) image and video quality (I/VQA) metrics with the possibility of using image metrics on videos framewise with averaging.

Written on **Python** and **PyTorch**. **52** methods have been implemented.

Most implementations are based on [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) and [PIQ](https://github.com/photosynthesis-team/piq). Some are taken from the repositories of the authors (see [List of available models](#list-of-available-models)). The VMAF implementation was taken from FFMPEG. 

# Table of Contents

* [Dependencies](#dependencies)
* [Installation](#installation)
	* [Docker](#docker)
* [How it works](#how-it-works)
	* [General](#general)
	* [Configs](#configs)
	* [Worklists](#worklists)
* [Example structure of work space](#example-structure-of-work-space)
* [Structure of config files](#structure-of-config-files)
	* [Common part](#common-part-for-image-and-video-datasets)
	* [Image Dataset](#image-dataset)
	* [Video Dataset](#video-dataset)
* [Structure of Worklists](#structure-of-worklists)
	* [Image Dataset](#image-dataset-1)
	* [Video Dataset](#video-dataset-1)
* [List of available models](#list-of-available-models)
	* [Image models](#image-models)
	* [Video models](#video-models)
* [Technical details](#technical-details)
	* [About VMAF](#about-vmaf)
* [Calculation pipeline](#calculation-pipeline)
* [License](#license)
* [Acknowledgement](#acknowledgement)

# Dependencies

* Python: >=3.10,<3.11
* [ffmpeg](https://ffmpeg.org/) (build with libvmaf for VMAF)
* [decord](https://github.com/dmlc/decord) (you can build decord with GPU to use NVDEC)
* [CUDA](https://developer.nvidia.com/cuda-toolkit): >= 10.2 (OPTIONAL if use GPU)
* [CuPy](https://docs.cupy.dev/en/stable/index.html) (OPTIONAL if use SI, CF, TI with GPU)

# Installation

**Via pip from PyPi:**

```
$ pip install objective-metrics
```

## Docker

Read [here](./Docker/README.md) about Docker support.
Dockerfiles are in `./Docker/` folder.

# How it works

## General

When you install this package you will have acces to `objective_metrics_run` command line tool.

It's designed for running some of image/video quality metrics on a **dataset**.

Results are given in **csv** files (format: `filename, metric value`) for each dataset and metric.

Models for extracting features from pretrained classificators are also implemented (currently only InceptionV3), results are given in **npz** files.

Name of resulting csv file for each dataset and metric: `{dataset_name}_{metric}.csv`

Name of resulting npz file for each dataset: `{dataset_name}_{feature_extractor}-features.npz`

## Configs

To use this tool, you have to make a **config** (yaml file) file for each dataset that you want to calculate.

Config files should be placed in ```./datasets_configs``` directory.

Tool takes dataset names as positional arguments. Each name should **exactly** match one config file from ```./datasets_configs``` directory.

## Worklists

Also, for each dataset you should have a ```{config_filename}_list.json``` file for **NR metrics** and **feature extractors** and ```{config_filename}_pairs.json``` for **FR metrics**.

Both files also should be placed in ```./datasets_configs``` directory and **exactly** match one config file (yaml) and tool positional argument.

><ins>Note:</ins> name of dataset in tool positional argument (and in the names of files with configs) may differ from **dataset_name** field in config (which only affects on the name of files with results and logs).

<ins>NR Worklist:</ins> ```{config_filename}_list.json``` is simply a list of images/videos to compute.


<ins>FR Worklist:</ins> ```{config_filename}_pairs.json``` is a dictionary with reference-distorted pairs.

You can see examples of configfs in **dataset_configs** folder.

More details about all config files are provided below.

# Example structure of work space:

```
dataset_configs/
├─ ...
├─ live-wcvqd_list.json  
├─ live-wcvqd_pairs.json  
├─ live-wcvqd.yaml  
├─ ...
├─ tid2013_list.json  
├─ tid2013_pairs.json  
├─ tid2013.yaml
├─ ...  
├─ live-vqc_list.json
├─ live-vqc.yaml  
├─ ...
```

`$ objective_metrics_run live-wcvqd tid2013 `

This command will run pipeline on two datasets: **live-wcvqd** and **tid2013**.


# Structure of config files:

## Common part for image and video datasets

* `type` - img/vid
* `dataset_pth` - path to dataset folder
* `dataset_name` - dataset name for resulting files (ex. {dataset_name}_{metric}.csv)
* `output` - name of directory with resulting files
* `logs_output` - name of file, where logs will be written (also will be created log file with process number in name)

## Image Dataset

<ins>Example tid2013.yaml:</ins>

```
type: img
dataset_pth: {PTH}/TID2013
dataset_name: TID2013
output: values
logs_output: objective_metrics_run.log
fr_metrics:
  - psnr
  - pieapp
nr_metrics:
  - koncept512
  - unique
feature_extractors:
  - inception_v3

```

* `fr_metrics` - list of FR IQA metrics to compute
* `nr_metrics` - list of NR IQA metrics to compute
* `feature_extractors` - list of image feature extractors (only inception_v3 is available)

## Video Dataset

><ins>Note:</ins> Image models are computed frame-wise with further averaging.

<ins>Example live-wcvqd.yaml:</ins>

```
type: vid
dataset_pth: {pth}/LIVE_WCVQD
dataset_name: LIVE-WCVQD
output: values
logs_output: objective_metrics_run.log
fr_iqa_metrics:
  - ahiq
  - lpips
nr_iqa_metrics:
  - niqe
  - koncept512
fr_vqa_metrics:
  - vmaf
nr_vqa_metrics:
  - mdtvsfa
  - dover
image_feature_extractors:
  - inception_v3
```

* `fr_iqa_metrics` - list of FR IQA metrics to compute
* `nr_iqa_metrics` - list of NR IQA metrics to compute
* `fr_vqa_metrics` - list of FR VQA metrics to compute
* `nr_vqa_metrics` - list of NR VQA metrics to compute
* `image_feature_extractors` - list of image feature extractors (only inception_v3 available)

# Structure of Worklists

## Image Dataset

### NR Worklist

<ins>Name:</ins> {config_filename}_list.json (ex. tid2013_list.json)

It is mandatory if you want to use **NR metrics** and **feature extractors**. File should be in **json** format with the following structure:

```
['img_pth_1', ...,'img_pth_n']
```

All paths should be relative to dataset_pth specified in yaml config.

### FR Worklist

<ins>Name:</ins> {config_filename}_pairs.json (ex. tid2013_pairs.json)

It is mandatory if you want to use **FR metrics**. File should be in .json format with the following structure:

```
{'ref_img_pth_1': ['dist_img_pth_1', ...,'dist_img_pth_n'], ..., 'ref_img_pth_n': [...]}

```

One record corresponds to one reference image and the list of its distortions.

All paths should be relative to dataset_pth specified in yaml config.

## Video Dataset

### NR Worklist

<ins>Name:</ins> {config_filename}_list.json (ex. live-wcvqd_list.json)

It is mandatory if you want to use **NR metrics** and **feature extractors**. File should be in .json format with the following structure:

```
['video_pth_1', ...,'video_pth_n']
```

All paths should be relative to dataset_pth specified in yaml config.

### FR Worklist

<ins>Name:</ins> {config_filename}_pairs.json (ex. live-wcvqd_pairs.json)

It is mandatory if you want to use **FR metrics**. File should be in .json format with the following structure:

`
Dict[str, List[Tuple[List[str], Literal["same", "repeat_each", "repeat_last"]]]]
`

```
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

```

One record corresponds to one reference video and the list of lists of its distortions.

All paths should be relative to dataset_pth specified in yaml config.

Algortihm will iterate along videos from one tuple simultaneously untill the shortest of them will end.

***
<ins>Example:</ins>

	{
		ref: 
			[
				([dist_1, dist_2, dist_long], "same"),
				([dist_3, dist_4], "repeat_each")
			]
	}

Here dist_1 and dist_2 have the same length that is also equal to ref length.

dist_long is longer than dist_1 and dist_2, so the excess frames of dist_long will be dropped and you will see warning in log file.

dist_3 and dist_4 have the same length, but it's n times shorter than the ref video length. So, to match the ref length, each frame of dist_3 and dist_4 will be taken n times. If n isn't an integer then frames to repeat will be **evenly spaced**.
***

Algorithm will compute frame to frame scores untill one of the videos will end.
If one of the source or distorted videos is shorter, then warning will be thrown in the log file.

If number of frames in the reference video is greater then in distorted videos from one record (may be beacuse of frame dropping distortion), then you can provide **mode** for handling this situation. All videos from one record will be treated identically.

<ins>mode:</ins>

* `same` -  iterate untill the shortest will end

* `repeat_each` - if the shortest distorted video is **n** times shorter than reference, then each frame of the shortest distorted video and corresponding frames of other distorted videos will be repeated **n** times. If **n** isn't an integer then frames to repeat will be evenly spaced. It works almost like ffmpeg **framerate filter**, except that after **repeat_each** you will always have the same lengths for ref and dist

* `repeat_last` - in all distorted videos last frame will be repeated as many times as needed to the shortest distorted video match the reference

If source video is shorter, then use mode == "same" to simply drop excess frames of distorted videos.
That's may be suitable, for example, when distorted videos have stalls.

<ins>Recommendation</ins>: this config has been designed in such a way that it is convenient to group the videos with same length in one tuple and apply one processing method to them. It also speeds up the calculation process.

# List of available models

>Names of models for config files are in brackets.

## Image models

### NR IQA

| Paper Link | Method | Code |
| ----------- | ---------- | ------------|
| [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)| SPAQ Baseline (spaq-bl) | [PyTorch](https://github.com/h4nwei/SPAQ) |
| [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)| SPAQ MT-A (spaq-mta)    | [PyTorch](https://github.com/h4nwei/SPAQ) |
| [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf)| SPAQ MT-S (spaq-mts)    | [PyTorch](https://github.com/h4nwei/SPAQ) |
| [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)     | HyperIQA (hyperiqa)      | [PyTorch](https://github.com/SSL92/hyperIQA) |
| [pdf](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) | CNNIQA (cnniqa) | [PyTorch](https://github.com/chaofengc/IQA-PyTorch) |
| [arXiv](https://arxiv.org/abs/2008.03889) | Linearity (linearity)      | [PyTorch](https://github.com/lidq92/LinearityIQA)              |
| [arXiv](https://arxiv.org/abs/1912.10088) | PaQ2PiQ (paq2piq)       | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/2207.12396) | CLIPIQA (clipiqa)       | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/2207.12396) | CLIPIQA+ (clipiqa+)      | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/1910.06180) | KonCept512 (koncept512)     | [PyTorch](https://github.com/ZhengyuZhao/koniq-PyTorch)        |
| [arXiv](https://arxiv.org/abs/2204.08958) | MANIQA (maniqa)        | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/2108.06858) | TReS (tres)          | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/2108.05997) | MUSIQ (musiq)         | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/1809.07517) | PI (pi)            | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/1907.02665) | DBCNN (dbcnn)         | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [arXiv](https://arxiv.org/abs/1709.05424) | NIMA (nima) | [PyTorch](https://github.com/titu1994/neural-image-assessment) |
| [arXiv](https://arxiv.org/abs/1612.05890) | NRQM (nrqm)          | [PyTorch](https://github.com/chaofengc/IQA-PyTorch)            |
| [pdf](https://live.ece.utexas.edu/publications/2015/zhang2015feature.pdf) | ILNIQE (ilniqe) | [PyTorch](https://github.com/chaofengc/IQA-PyTorch) |
| [pdf](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf)    | BRISQUE (brisque) | [PyTorch](https://github.com/chaofengc/IQA-PyTorch) |
| [pdf](https://live.ece.utexas.edu/publications/2013/mittal2013.pdf)       | NIQE (niqe)    | [PyTorch](https://github.com/chaofengc/IQA-PyTorch) |
[arXiv](https://arxiv.org/abs/2005.13983) | UNIQUE (unique) |[PyTorch](https://github.com/chaofengc/IQA-PyTorch) |
[arXiv](https://arxiv.org/abs/2308.03060)| TOPIQ (topiq_nr) | [PyTorch](https://github.com/chaofengc/IQA-PyTorch) |
[ITU](https://www.itu.int/rec/T-REC-P.910)| Spatial Information (si) | self-made | No | - |
[ResearchGate](https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images)| Colourfulness (cf) | self-made |


### FR IQA

>PSNR, SSIM, MS-SSIM, CW-SSIM are computed on Y channel in YUV (YCbCr) color space.

| Paper Link | Method | Code |
| ----------- | ---------- | ------------|
| [arXiv](https://arxiv.org/abs/2308.03060) | TOPIQ (topiq_fr)   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://arxiv.org/abs/2204.10485) | AHIQ (ahiq)   	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://arxiv.org/abs/1806.02067) | PieAPP (pieapp)    | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://arxiv.org/abs/1801.03924) | LPIPS (lpips)  	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://arxiv.org/abs/2004.07728) | DISTS (dists)  	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://arxiv.org/abs/2108.07948) | CKDN<sup>[1](#fn1)</sup> (ckdn)   	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [pdf](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf) | FSIM (fsim)   	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [wiki](https://en.wikipedia.org/wiki/Structural_similarity) | SSIM (ssim)    	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [pdf](https://www.researchgate.net/publication/2931584_Multi-Scale_Structural_Similarity_for_Image_Quality_Assessment) | MS-SSIM (ms_ssim)  | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [pdf](https://live.ece.utexas.edu/publications/2009/sampat_tip_nov09.pdf) | CW-SSIM (cw_ssim)  | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)| PSNR (psnr)   	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [pdf](https://live.ece.utexas.edu/publications/2004/hrs_ieeetip_2004_imginfo.pdf)| VIF (vif)	  		   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [arXiv](https://arxiv.org/abs/1308.3052) | GMSD (gmsd)	  	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [pdf](https://www.uv.es/lapeva/papers/2016_HVEI.pdf) | NLPD (nlpd)	  	   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [IEEE Xplore](https://ieeexplore.ieee.org/document/6873260)| VSI (vsi)	  		   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [pdf](https://www.researchgate.net/publication/220050520_Most_apparent_distortion_Full-reference_image_quality_assessment_and_the_role_of_strategy) | MAD (mad)	  		   | [PyTorch](https://github.com/chaofengc/IQA-PyTorch/tree/main) |
| [IEEE Xplore](https://ieeexplore.ieee.org/document/6467149) | SR-SIM (srsim)     | [PyTorch](https://github.com/photosynthesis-team/piq?tab=readme-ov-file) |
| [IEEE Xplore](https://ieeexplore.ieee.org/document/7351172)| DSS (dss)	  		   | [PyTorch](https://github.com/photosynthesis-team/piq?tab=readme-ov-file) |
| [arXiv](https://arxiv.org/abs/1607.06140)| HaarPSI (haarpsi)  | [PyTorch](https://github.com/photosynthesis-team/piq?tab=readme-ov-file) |
| [arXiv](https://arxiv.org/abs/1608.07433) | MDSI (mdsi) 		   | [PyTorch](https://github.com/photosynthesis-team/piq?tab=readme-ov-file) |
| [pdf](https://www.researchgate.net/publication/317724142_Gradient_magnitude_similarity_deviation_on_multiple_scales_for_color_image_quality_assessment) | MS-GMSD (msgmsd)	   | [PyTorch](https://github.com/photosynthesis-team/piq?tab=readme-ov-file) |

<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
 
### Feature Extractors

| Paper Link | Method | Code |
| ----------- | ---------- | ------------|
| [arXiv](https://arxiv.org/abs/1512.00567) | InceptionV3 (inception_v3)   | [PyTorch](https://pytorch.org/vision/stable/index.html) |

## Video models

### NR VQA


| Paper Link | Method | Code |
| ----------- | ---------- | ------------|
| [arXiv](https://arxiv.org/abs/2011.04263) | MDTVSFA (mdtvsfa) | [PyTorch](https://github.com/lidq92/MDTVSFA)               |
| [arXiv](https://arxiv.org/abs/2207.02595) | FAST-VQA (FAST-VQA) | [PyTorch](https://github.com/teowu/FAST-VQA-and-FasterVQA) |
| [arXiv](https://arxiv.org/abs/2207.02595) | FasterVQA (FasterVQA) | [PyTorch](https://github.com/teowu/FAST-VQA-and-FasterVQA) |
| [arXiv](https://arxiv.org/abs/2211.04894) | DOVER (dover)    | [PyTorch](https://github.com/VQAssessment/DOVER)           |
| [ITU](https://www.itu.int/rec/T-REC-P.910) | Temporal Information (ti)    | self-made |

### FR VQA

| Paper Link | Method | Code |
| ----------- | ---------- | ------------|
| [wiki](https://en.wikipedia.org/wiki/Video_Multimethod_Assessment_Fusion) | VMAF (vmaf) | [FFMPEG VMAF](https://github.com/Netflix/vmaf) |

# Technical details

0. Application firstly check if CUDA is available (and if CuPy is installed for SI and CF). If true, then GPU is used. If you don't want to use GPU, set enviroment variable `CUDA_VISIBLE_DEVICES=-1`

1. All necessary checkpoints for models will be automatically downloaded in $HOME/.cache/{...}. All real paths will be written in log files

2. Application is always trying to **not to stop** because of errors and process them on the go with logging to log files

3. All csv files with metric values and logs are written **interactively** line by line

4. Images and frames of videos are rescaled to **one size** using bicubic interpolation before computing FR metrics

5. IQA metrics and image feature extractors for videos are computed **frame-wise** with further averaging

	1. If some of frames can't be computed during loop - they will be **thrown away** with warning in log file

6. All one type (NR or FR) IQA metrics are computed in one pass

	1. For images and videos, this means that they will be read only once for all listed in config one type metrics

	2. Also, all distorted images/videos from one set in FR Worklist will be computed simultaneously, which reduces the reading of reference to one time

	3. Because of this, it is necessary to keep all initialized models in memory, so be careful with cuda when writing configs

7. If you have build Decord with GPU support and want to use NVDEC, set enviroment variable `DECORD_GPU=1`

8. The application first tries to read videos using the Decord with GPU acceleration (if `DECORD_GPU=1`). Next, if it didn't work out (if Decord threw an exception), with Decord on CPU. If this also failed (if Decord threw an exception), then with the help of OpenCV on CPU

	1. Build Decord with GPU acceleration (with NVDEC) to make reading videos from memory faster

9. If you don't want to use Decord at all and only want OpenCV, set enviroment variable `USE_CV2=1`

10. If you want to see the time of counting each frame in the video with image models in logs, set enviroment variable `DEBUG_VIDEO=1`. It will also make ffmpeg VMAF output visible in logs

## **About VMAF**

For the sake of generality, VMAF in this package is using the same config file and same modes, as needed for FR Worklist: {config_filename}_pairs.json.

There is no need to group the videos with the same length. For one reference you can just make one or two groups (tuples) with different mode.

Each pair of videos will be processed individually.

For proper work, when computing VMAF, both videos will be brought to one framerate (reference) with ffmpeg **framerate filter**.

After that, if videos still have different length - two ffmpeg **framesync** options are available: `shortest` and `repeatlast`.

Modes `same` and `repeat_each` will be treated as `shortest`.

Mode `repeat_last` will be treated as `repeatlast`.

To specify number of threads, set enviroment variable `VMAF_N_THREADS`. (Default: 1)

# Calculation pipeline

## 0. Transmuxing

Transmux your videos from yuv format to some container with metadata (mp4, webm, mkv, y4m, ...). You can use lossles encoding.

## 1. Make configs for your datasets

### a. Worklists


If you want to calculate NR metrics and/or extract features, then make NR Worklist **{config_filename}_list.json**.

If you want to calculate FR metrics, then make FR Worklist **{config_filename}_pairs.json**. 

That is the most complicated part, where you shold specify all videos that you want to be computed.

You can find some examples of worklists in **datasets_configs/** folder.

### b. Config

Make config file **{dataset_name}.yaml**, where write all necessary information, like path to dataset. 

There you can write a list of metrics, that you want to compute over the dataset. 

You can find examples of configs in **datasets_configs/** folder.

## 2. Run application

Run objective_metrics_run script from command line.

You also can build a docker image and run a container. Read [here](./Docker/README.md) about Docker support. Dockerfiles are in **./Docker/** folder.

## 3. Watch the logs and values

### a. Logs

Application will make two log files and will be writing to them interactively.

One log file will be called the same as the number of process in which application was started (in case of docker image it will be just 1.log). This log file will contain general information about queue of datasets.

Second log file will be named as a name of a dataset, that is now being calculated. This log file will contain information about all steps of calculation and about all errors.

### b. Values

Application will make a csv file for each metric and dataset and will be writing to them interactively.

You can watch metric values in the corresponding csv files. If due to errors, for this concrete image or video the metric could not be calculated, then the value in the file will be Nan.

## 4. Error handling

Most common error is **CUDA out of memory**.

You can parse log and csv files to define what images and videos and what metrics failed to be calculated.

You can look examples of log parsers in **log_parsers.py** file.

After you defined all failed images/videos and metrics - you can make new worklists and configs to try again, when you will have more free CUDA memory available.

For example, if due to CUDA out of memory error for some metric more then 10 frames of video failed to be calculated, then this is an excuse to make another attmept for this video for this metric.

# License

This project is licensed under the MIT License. However, it also includes code distributed under the BSD+Patent license. See LICENSE file.

# Acknowledgement

[VQMT](https://compression.ru/video/quality_measure/vqmt_download.html) - Tool with objective metrics

[PyIQA](https://github.com/chaofengc/IQA-PyTorch/tree/main) - Python Library

[PIQ](https://github.com/photosynthesis-team/piq?tab=readme-ov-file) - Python Library

[VMAF](https://github.com/Netflix/vmaf)

[Table of Metrics](https://github.com/chaofengc/Awesome-Image-Quality-Assessment)

[MSU Metric Benchmark](https://videoprocessing.ai/benchmarks/video-quality-metrics.html)