"""
FAST-VQA and FasterVQA class.

src: https://github.com/teowu/FAST-VQA-and-FasterVQA
"""
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Literal
from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu


from .models._fastvqa import DiViDeAddEvaluator
from .utils._fastvqa import FragmentSampleFrames, SampleFrames, get_spatial_fragments, sigmoid_rescale, download_checkpoints, opts


class FastVQA:
    def __init__(self, metric: Literal["FAST-VQA", "FasterVQA"], logger: logging.Logger) -> None:
        """Init FastVQA model.
        """
        self.metric = metric
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        opt = opts.get(self.metric, opts["FAST-VQA"])
        opt = str(Path(__file__).parents[0] / "configs" / opt)
        with open(opt, "r") as f:
            opt = yaml.safe_load(f)

        ckpt_pth = download_checkpoints(self.metric, logger)

        self.evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(self.device)
        self.evaluator.load_state_dict(
            torch.load(ckpt_pth, map_location=self.device)["state_dict"]
        )

        self.t_data_opt = opt["data"]["val-kv1k"]["args"]
        self.s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]

    def score_video(self, video_pth: str, read_video: VideoReaderDecord | VideoReaderOpenCVcpu) -> float:
        """Score one video.

        Parameters
        ----------
        video_pth : str
            path to video.
        read_video : VideoReaderDecord | VideoReaderOpenCVcpu
            video reader

        Returns
        -------
        float
            score.
        """
        video_reader = read_video(video_pth)

        with torch.no_grad():
            vsamples = {}
            for sample_type, sample_args in self.s_data_opt.items():
                ## Sample Temporally
                if self.t_data_opt.get("t_frag", 1) > 1:
                    sampler = FragmentSampleFrames(
                        fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                        fragments_t=sample_args.get("t_frag", 1),
                        num_clips=sample_args.get("num_clips", 1),
                    )
                else:
                    sampler = SampleFrames(
                        clip_len=sample_args["clip_len"], num_clips=sample_args["num_clips"]
                    )

                num_clips = sample_args.get("num_clips", 1)
                frames = sampler(len(video_reader))
                frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
                imgs = [torch.tensor(np.array(frame_dict[idx])) for idx in frames]
                video = torch.stack(imgs, 0)
                video = video.permute(3, 0, 1, 2)

                ## Sample Spatially
                sampled_video = get_spatial_fragments(video, **sample_args)
                mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor(
                    [58.395, 57.12, 57.375]
                )
                sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(
                    3, 0, 1, 2
                )

                sampled_video = sampled_video.reshape(
                    sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
                ).transpose(0, 1)
                vsamples[sample_type] = sampled_video.to(self.device)

            result = self.evaluator(vsamples)
            score = sigmoid_rescale(result.mean().item(), model=self.metric)

        return score
