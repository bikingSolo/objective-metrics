import cv2
from decord import VideoReader, cpu, gpu, bridge
from PIL import Image
from typing import Iterator, Literal, Tuple


def get_metadata(vid) -> Tuple[float, int, int]:
    """Return fps, height and width of a video.

    Parameters
    ----------
    vid : str
        path to video
    Returns
    -------
    Tuple[float, int, int]
        fps, height, width
    """
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()

    return round(fps, 2), int(height), int(width)


class VideoReaderOpenCVcpu:
    """OpenCV cpu Video Reader. Behaves both like iterator and like list independently.
    
    Frames are presented as PIL IMages in the RGB with [0, 255] range.
    """
    def __init__(self, video_path: str) -> None:
        """Init Video Reader.

        Parameters
        ----------
        video_path : str
            path to video file.
        """
        self.video_path = video_path
        self.cap_iter = cv2.VideoCapture(self.video_path)
        self.cap_list = cv2.VideoCapture(self.video_path)
        self.len = None

    def __iter__(self) -> Iterator[Image.Image]:
        return self

    def __next__(self) -> Image:
        ret, frame = self.cap_iter.read()
        if ret == False:
            self.cap_iter.release()
            raise StopIteration
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    
    def __len__(self) -> int:
        if self.len is None:
            cap = cv2.VideoCapture(self.video_path)
            try:
                self.len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.len is None:
                    raise Exception
            except:
                self.len = 0
                while(cap.isOpened()):
                    ret, _ = cap.read()
                    if ret == False:
                        break
                    self.len += 1
            cap.release()
        return self.len
    
    def __getitem__(self, index: int) -> Image:
        self.cap_list.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap_list.read()
        if ret == False:
            raise IndexError(f"Index: {index} out of bound: {self.__len__()}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    
    def __del__(self) -> None:
        self.cap_iter.release()
        self.cap_list.release()


class VideoReaderDecord:
    """Decord Video Reader. Behaves both like iterator and like list independently.

    Frames are presented as PIL IMages in the RGB with [0, 255] range.
    """
    def __init__(self, video_path: str, mode: Literal["cpu", "gpu"] = "cpu") -> None:
        """Init Video Reader.

        Parameters
        ----------
        video_path : str
            path to video file.
        mode : Literal["cpu", "gpu"]
            if "gpu" then NVDEC used for acceleration.
            Only possible when Decord build from source with gpu support. 
        """
        bridge.set_bridge("native")
        self.video_path = video_path
        self.vr = VideoReader(video_path, cpu(0) if mode == "cpu" else gpu(0))
        self.index = 0
        self.len = len(self.vr)

    def __iter__(self) -> Iterator[Image.Image]:
        return self

    def __next__(self) -> Image:
        if self.index >= self.len:
            raise StopIteration
        frame = Image.fromarray(self.vr[self.index].asnumpy())
        self.index += 1
        return frame
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int) -> Image:
        return Image.fromarray(self.vr[index].asnumpy())
