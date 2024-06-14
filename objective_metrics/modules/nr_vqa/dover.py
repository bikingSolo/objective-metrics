"""
DOVER class.

src: https://github.com/VQAssessment/DOVER
"""
import logging
from pathlib import Path
import torch
import yaml
from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu

from .models._dover import DOVER as DVR
from .utils._dover import UnifiedFrameSampler, spatial_temporal_view_decomposition_v2, fuse_results, download_checkpoints, mean, std


class DOVER:
    def __init__(self, logger: logging.Logger) -> None:
        """Init DOVER model.
        """
        self.metric = "dover"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        conf_pth = str(Path(__file__).parents[0] / "configs/dover/dover.yml")

        with open(conf_pth, "r") as f:
            opt = yaml.safe_load(f)

        ckpt_pth = download_checkpoints(self.metric, logger)

        self.evaluator = DVR(**opt["model"]["args"]).to(self.device)
        self.evaluator.load_state_dict(
            torch.load(ckpt_pth, map_location=self.device)
        )

        self.dopt = opt["data"]["val-l1080p"]["args"]

        self.temporal_samplers = {}
        for stype, sopt in self.dopt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

    def score_video(self, video_pth: str, video_reader: VideoReaderDecord | VideoReaderOpenCVcpu) -> float:
        """Score one video.

        Parameters
        ----------
        video_pth : str
            path to video.
        video_reader : VideoReaderDecord | VideoReaderOpenCVcpu
            video reader

        Returns
        -------
        float
            score
        """

        views, _ = spatial_temporal_view_decomposition_v2(
            video_pth, self.dopt["sample_types"], self.temporal_samplers, video_reader
        )

        with torch.no_grad():
            for k, v in views.items():
                num_clips = self.dopt["sample_types"][k].get("num_clips", 1)
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(self.device)
                )

            score = fuse_results([r.mean().item() for r in self.evaluator(views)])

        return score
