"""
MDTVSFA class.

src: https://github.com/lidq92/MDTVSFA
"""
import torch
import logging
from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu

from .models._mdtvsfa import VQAModel
from .utils._mdtvsfa import get_features_mod_3, download_checkpoints, FRAME_BATCH_SIZE


class MDTVSFA:
    def __init__(self, logger: logging.Logger = None) -> None:
        """Init MDTVSFA model.
        """
        self.metric = "mdtvsfa"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQAModel().to(self.device)
        ckpt_pth = download_checkpoints(self.metric, logger)
        self.model.load_state_dict(torch.load(ckpt_pth, map_location=self.device))
        self.model.eval()
        self.frame_batch_size = FRAME_BATCH_SIZE

    def score_video(self, video_pth: str, video_reader: VideoReaderDecord | VideoReaderOpenCVcpu) -> float:
        """Score one video.

        Parameters
        ----------
        video_pth : str
            path to video
        video_reader : VideoReaderDecord | VideoReaderOpenCVcpu
            video reader

        Returns
        -------
        float
            score
        """
        with torch.no_grad():
            features = get_features_mod_3(
                video_pth, video_reader, frame_batch_size=self.frame_batch_size, device=self.device
            )
            features = torch.unsqueeze(features, 0)

            input_length = features.shape[1] * torch.ones(1, 1, dtype=torch.long)
            _, mapped_score, _ = self.model([(features, input_length, ["K"])])
            score = mapped_score[0][0].item()

        return score
