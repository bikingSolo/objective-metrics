"""
TI module.
"""
import torch
from ..utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu


class TI:
    def __init__(self) -> None:
        """Init TI module.
        """
        if torch.cuda.is_available():
            try:
                from .models._ti_gpu import ti_gpu
                self.model = ti_gpu
            except:
                from .models._ti import ti_cpu
                self.model = ti_cpu
        else:
            from .models._ti import ti_cpu
            self.model = ti_cpu
        self.metric = "ti"

    def score_video(self, video_pth: str, video_reader: VideoReaderDecord | VideoReaderOpenCVcpu) -> float:
        """Score one video via averaging by frames.

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
        reader = video_reader(video_pth)

        ti_lst = []
        prev_frame = None
        for frame in reader:
            if prev_frame is not None:
                ti_lst.append(self.model(frame, prev_frame))
            prev_frame = frame

        return sum(ti_lst) / len(ti_lst)
