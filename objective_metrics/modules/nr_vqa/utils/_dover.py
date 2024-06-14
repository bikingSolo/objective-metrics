import os
import logging
import time
import gdown
import os.path as osp
import random
from functools import lru_cache
from pathlib import Path

import cv2
import decord
import numpy as np
import skvideo.io
import torch
import torchvision
from decord import VideoReader
from tqdm import tqdm

random.seed(42)

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

## by Leo Borisovsky.
def download_checkpoints(metric: str, logger: logging.Logger) -> str:
    """
    Returns
    -------
    str
        path to checkpoint.
    """
    ckpts_dir = Path.home() / ".cache" / "objective_metrics" / "checkpoints"
    dover_ckpts_dir = ckpts_dir / "DOVER"
    dover_ckpt_pth = dover_ckpts_dir / "DOVER.pth"

    os.makedirs(dover_ckpts_dir, exist_ok=True)

    if not os.path.isfile(dover_ckpt_pth):
        start = time.time()
        if logger is not None:
            logger.info(f"Downloading checkpoints for {metric} to {dover_ckpts_dir} ...")
        gdown.download(
            "https://drive.google.com/file/d/1LjSr3OPR8mBsEm-Bc7qFpUJ9XEdZtDRG/view?usp=drive_link",
            fuzzy=True, output=str(dover_ckpt_pth)
        )
        if logger is not None:
            logger.info(f"Done in {time.time() - start} seconds")
    elif logger is not None:
        logger.info(f"Initializing {metric} with checkpoints from {dover_ckpts_dir}.")

    return str(dover_ckpt_pth)


def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    print(x)
    return 1 / (1 + np.exp(-x))


def gaussian_rescale(pr):
    # The results should follow N(0,1)
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr


def uniform_rescale(pr):
    # The result scores should follow U(0,1)
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)


## Basic Dataset
def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    fallback_type="upsample",
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w

    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video


class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1):
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def get_frame_indices(self, num_frames):
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )
        return np.concatenate(ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []
        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]
        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds


class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)


class FastVQAPlusPlusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        frame_interval=2,
        aligned=32,
        fragments=(8, 8, 8),
        fsize=(4, 32, 32),
        num_clips=1,
        nfrags=1,
        cache_in_memory=False,
        phase="test",
        fallback_type="oversample",
    ):
        """
        Fragments.
        args:
            fragments: G_f as in the paper.
            fsize: S_f as in the paper.
            nfrags: number of samples (spatially) as in the paper.
            num_clips: number of samples (temporally) as in the paper.
        """
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.clip_len = fragments[0] * fsize[0]
        self.aligned = aligned
        self.fallback_type = fallback_type
        self.sampler = FragmentSampleFrames(
            fsize[0], fragments[0], frame_interval, num_clips
        )
        self.video_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self,
        index,
        tocache=False,
        need_original_frames=False,
    ):
        decord.bridge.set_bridge("torch")
        if tocache or self.cache is None:
            fx, fy = self.fragments[1:]
            fsx, fsy = self.fsize[1:]
            video_info = self.video_infos[index]
            filename = video_info["filename"]
            label = video_info["label"]
            if filename.endswith(".yuv"):
                video = skvideo.io.vread(
                    filename, 1080, 1920, inputdict={"-pix_fmt": "yuvj420p"}
                )
                frame_inds = self.sampler(video.shape[0], self.phase == "train")
                imgs = [torch.from_numpy(video[idx]) for idx in frame_inds]
            else:
                vreader = VideoReader(filename)
                frame_inds = self.sampler(len(vreader), self.phase == "train")
                frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
                imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            if self.nfrags == 1:
                vfrag = get_spatial_fragments(
                    video,
                    fx,
                    fy,
                    fsx,
                    fsy,
                    aligned=self.aligned,
                    fallback_type=self.fallback_type,
                )
            else:
                vfrag = get_spatial_fragments(
                    video,
                    fx,
                    fy,
                    fsx,
                    fsy,
                    aligned=self.aligned,
                    fallback_type=self.fallback_type,
                )
                for i in range(1, self.nfrags):
                    vfrag = torch.cat(
                        (
                            vfrag,
                            get_spatial_fragments(
                                video,
                                fragments,
                                fx,
                                fy,
                                fsx,
                                fsy,
                                aligned=self.aligned,
                                fallback_type=self.fallback_type,
                            ),
                        ),
                        1,
                    )
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "video": vfrag.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + vfrag.shape[2:]
            ).transpose(
                0, 1
            ),  # B, V, T, C, H, W
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_video"] = video.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + video.shape[2:]
            ).transpose(0, 1)
        return data

    def __len__(self):
        return len(self.video_infos)


class FragmentVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        aligned=32,
        fragments=7,
        fsize=32,
        nfrags=1,
        cache_in_memory=False,
        phase="test",
    ):
        """
        Fragments.
        args:
            fragments: G_f as in the paper.
            fsize: S_f as in the paper.
            nfrags: number of samples as in the paper.
        """
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.aligned = aligned
        self.sampler = SampleFrames(clip_len, frame_interval, num_clips)
        self.video_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self,
        index,
        fragments=-1,
        fsize=-1,
        tocache=False,
        need_original_frames=False,
    ):
        decord.bridge.set_bridge("torch")
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            video_info = self.video_infos[index]
            filename = video_info["filename"]
            label = video_info["label"]
            if filename.endswith(".yuv"):
                video = skvideo.io.vread(
                    filename, 1080, 1920, inputdict={"-pix_fmt": "yuvj420p"}
                )
                frame_inds = self.sampler(video.shape[0], self.phase == "train")
                imgs = [torch.from_numpy(video[idx]) for idx in frame_inds]
            else:
                vreader = VideoReader(filename)
                frame_inds = self.sampler(len(vreader), self.phase == "train")
                frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
                imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            if self.nfrags == 1:
                vfrag = get_spatial_fragments(
                    video, fragments, fragments, fsize, fsize, aligned=self.aligned
                )
            else:
                vfrag = get_spatial_fragments(
                    video, fragments, fragments, fsize, fsize, aligned=self.aligned
                )
                for i in range(1, self.nfrags):
                    vfrag = torch.cat(
                        (
                            vfrag,
                            get_spatial_fragments(
                                video,
                                fragments,
                                fragments,
                                fsize,
                                fsize,
                                aligned=self.aligned,
                            ),
                        ),
                        1,
                    )
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "video": vfrag.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + vfrag.shape[2:]
            ).transpose(
                0, 1
            ),  # B, V, T, C, H, W
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_video"] = video.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + video.shape[2:]
            ).transpose(0, 1)
        return data

    def __len__(self):
        return len(self.video_infos)


class ResizedVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        aligned=32,
        size=224,
        cache_in_memory=False,
        phase="test",
    ):
        """
        Using resizing.
        """
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.size = size
        self.aligned = aligned
        self.sampler = SampleFrames(clip_len, frame_interval, num_clips)
        self.video_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching resized videos"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(self, index, tocache=False, need_original_frames=False):
        decord.bridge.set_bridge("torch")
        if tocache or self.cache is None:
            video_info = self.video_infos[index]
            filename = video_info["filename"]
            label = video_info["label"]
            vreader = VideoReader(filename)
            frame_inds = self.sampler(len(vreader), self.phase == "train")
            frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
            imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            video = torch.nn.functional.interpolate(video, size=(self.size, self.size))
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "video": vfrag.reshape(
                (-1, self.num_clips, self.clip_len) + vfrag.shape[2:]
            ).transpose(
                0, 1
            ),  # B, V, T, C, H, W
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_video"] = video.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + video.shape[2:]
            ).transpose(0, 1)
        return data

    def __len__(self):
        return len(self.video_infos)


class CroppedVideoDataset(FragmentVideoDataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        aligned=32,
        size=224,
        ncrops=1,
        cache_in_memory=False,
        phase="test",
    ):
        """
        Regard Cropping as a special case for Fragments in Grid 1*1.
        """
        super().__init__(
            ann_file,
            data_prefix,
            clip_len=clip_len,
            frame_interval=frame_interval,
            num_clips=num_clips,
            aligned=aligned,
            fragments=1,
            fsize=224,
            nfrags=ncrops,
            cache_in_memory=cache_in_memory,
            phase=phase,
        )


class FragmentImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        fragments=7,
        fsize=32,
        nfrags=1,
        cache_in_memory=False,
        phase="test",
    ):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.image_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.image_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.image_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self, index, fragments=-1, fsize=-1, tocache=False, need_original_frames=False
    ):
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            image_info = self.image_infos[index]
            filename = image_info["filename"]
            label = image_info["label"]
            try:
                img = torchvision.io.read_image(filename)
            except:
                img = cv2.imread(filename)
                img = torch.from_numpy(img[:, :, [2, 1, 0]]).permute(2, 0, 1)
            img_shape = img.shape[1:]
            image = img.unsqueeze(1)
            if self.nfrags == 1:
                ifrag = get_spatial_fragments(image, fragments, fragments, fsize, fsize)
            else:
                ifrag = get_spatial_fragments(image, fragments, fragments, fsize, fsize)
                for i in range(1, self.nfrags):
                    ifrag = torch.cat(
                        (
                            ifrag,
                            get_spatial_fragments(
                                image, fragments, fragments, fsize, fsize
                            ),
                        ),
                        1,
                    )
            if tocache:
                return (ifrag, label, img_shape)
        else:
            ifrag, label, img_shape = self.cache[index]
        if self.nfrags == 1:
            ifrag = (
                ((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std)
                .squeeze(0)
                .permute(2, 0, 1)
            )
        else:
            ### During testing, one image as a batch
            ifrag = (
                ((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std)
                .squeeze(0)
                .permute(0, 3, 1, 2)
            )
        data = {
            "image": ifrag,
            "gt_label": label,
            "original_shape": img_shape,
            "name": filename,
        }
        if need_original_frames:
            data["original_image"] = image.squeeze(1)
        return data

    def __len__(self):
        return len(self.image_infos)


class ResizedImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        size=224,
        cache_in_memory=False,
        phase="test",
    ):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.size = size
        self.image_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.image_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.image_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self, index, fragments=-1, fsize=-1, tocache=False, need_original_frames=False
    ):
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            image_info = self.image_infos[index]
            filename = image_info["filename"]
            label = image_info["label"]
            img = torchvision.io.read_image(filename)
            img_shape = img.shape[1:]
            image = img.unsqueeze(1)
            if self.nfrags == 1:
                ifrag = get_spatial_fragments(image, fragments, fsize)
            else:
                ifrag = get_spatial_fragments(image, fragments, fsize)
                for i in range(1, self.nfrags):
                    ifrag = torch.cat(
                        (ifrag, get_spatial_fragments(image, fragments, fsize)), 1
                    )
            if tocache:
                return (ifrag, label, img_shape)
        else:
            ifrag, label, img_shape = self.cache[index]
        ifrag = (
            ((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std)
            .squeeze(0)
            .permute(2, 0, 1)
        )
        data = {
            "image": ifrag,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_image"] = image.squeeze(1)
        return data

    def __len__(self):
        return len(self.image_infos)


class CroppedImageDataset(FragmentImageDataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        size=224,
        ncrops=1,
        cache_in_memory=False,
        phase="test",
    ):
        """
        Regard Cropping as a special case for Fragments in Grid 1*1.
        """
        super().__init__(
            ann_file,
            data_prefix,
            fragments=1,
            fsize=224,
            nfrags=ncrops,
            cache_in_memory=cache_in_memory,
            phase=phase,
        )


## Fusion Dataset
def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video


@lru_cache
def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop(
            (size_h, size_w), scale=(0.40, 1.0)
        )
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))


def get_resized_video(
    video,
    size_h=224,
    size_w=224,
    random_crop=False,
    arp=False,
    **kwargs,
):
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(
        size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop
    )
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video


def get_arp_resized_video(
    video,
    short_edge=224,
    train=False,
    **kwargs,
):
    if train:  ## if during training, will random crop into square and then resize
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[..., rnd_h : rnd_h + ori_short_edge, :]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[..., :, rnd_h : rnd_h + ori_short_edge]
    ori_short_edge = min(video.shape[-2:])
    scale_factor = short_edge / ori_short_edge
    ovideo = video
    video = torch.nn.functional.interpolate(
        video / 255.0, scale_factors=scale_factor, mode="bilinear"
    )
    video = (video * 255.0).type_as(ovideo)
    return video


def get_arp_fragment_video(
    video,
    short_fragments=7,
    fsize=32,
    train=False,
    **kwargs,
):
    if (
        train
    ):  ## if during training, will random crop into square and then get fragments
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[..., rnd_h : rnd_h + ori_short_edge, :]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[..., :, rnd_h : rnd_h + ori_short_edge]
    kwargs["fsize_h"], kwargs["fsize_w"] = fsize, fsize
    res_h, res_w = video.shape[-2:]
    if res_h > res_w:
        kwargs["fragments_w"] = short_fragments
        kwargs["fragments_h"] = int(short_fragments * res_h / res_w)
    else:
        kwargs["fragments_h"] = short_fragments
        kwargs["fragments_w"] = int(short_fragments * res_w / res_h)
    return get_spatial_fragments(video, **kwargs)


def get_cropped_video(
    video,
    size_h=224,
    size_w=224,
    **kwargs,
):
    kwargs["fragments_h"], kwargs["fragments_w"] = 1, 1
    kwargs["fsize_h"], kwargs["fsize_w"] = size_h, size_w
    return get_spatial_fragments(video, **kwargs)


def get_single_view(
    video,
    sample_type="aesthetic",
    **kwargs,
):
    if sample_type.startswith("aesthetic"):
        video = get_resized_video(video, **kwargs)
    elif sample_type.startswith("technical"):
        video = get_spatial_fragments(video, **kwargs)
    elif sample_type == "original":
        return video

    return video


def spatial_temporal_view_decomposition(
    video_path,
    sample_types,
    samplers,
    is_train=False,
    augment=False,
):
    video = {}
    if video_path.endswith(".yuv"):
        print("This part will be deprecated due to large memory cost.")
        ## This is only an adaptation to LIVE-Qualcomm
        ovideo = skvideo.io.vread(
            video_path, 1080, 1920, inputdict={"-pix_fmt": "yuvj420p"}
        )
        for stype in samplers:
            frame_inds = samplers[stype](ovideo.shape[0], is_train)
            imgs = [torch.from_numpy(ovideo[idx]) for idx in frame_inds]
            video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)
        del ovideo
    else:
        decord.bridge.set_bridge("torch")
        vreader = VideoReader(video_path)
        all_frame_inds = []
        frame_inds = {}
        for stype in samplers:
            frame_inds[stype] = samplers[stype](len(vreader), is_train)
            all_frame_inds.append(frame_inds[stype])

        ### Each frame is only decoded one time!!!
        all_frame_inds = np.concatenate(all_frame_inds, 0)
        frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}

        for stype in samplers:
            imgs = [frame_dict[idx] for idx in frame_inds[stype]]
            video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_view(video[stype], stype, **sopt)
    return sampled_video, frame_inds


## Leo Borisovsky (24.02.2024). Support of different Video Readers.
from ...utils.video_readers import VideoReaderDecord, VideoReaderOpenCVcpu

def spatial_temporal_view_decomposition_v2(
    video_path,
    sample_types,
    samplers,
    read_video: VideoReaderDecord | VideoReaderOpenCVcpu,
    is_train=False,
):
    video = {}
    
    vreader = read_video(video_path)
    all_frame_inds = []
    frame_inds = {}
    for stype in samplers:
        frame_inds[stype] = samplers[stype](len(vreader), is_train)
        all_frame_inds.append(frame_inds[stype])

    all_frame_inds = np.concatenate(all_frame_inds, 0)
    frame_dict = {idx: torch.tensor(np.array(vreader[idx])) for idx in np.unique(all_frame_inds)}  ## modification because no more decord torch bridge

    for stype in samplers:
        imgs = [frame_dict[idx] for idx in frame_inds[stype]]
        video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_view(video[stype], stype, **sopt)
    return sampled_video, frame_inds


import random
import numpy as np


class UnifiedFrameSampler:
    def __init__(
        self,
        fsize_t,
        fragments_t,
        frame_interval=1,
        num_clips=1,
        drop_rate=0.0,
    ):
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )

        drop = random.sample(
            list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate)
        )
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)


class ViewDecompositionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        ## opt is a dictionary that includes options for video sampling

        super().__init__()

        self.weight = opt.get("weight", 0.5)

        self.video_infos = []
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.opt = opt
        self.sample_types = opt["sample_types"]
        self.data_backend = opt.get("data_backend", "disk")
        self.augment = opt.get("augment", False)
        if self.data_backend == "petrel":
            from petrel_client import client

            self.client = client.Client(enable_mc=True)

        self.phase = opt["phase"]
        self.crop = opt.get("random_crop", False)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}
        for stype, sopt in opt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
            print(
                stype + " branch sampled frames:",
                self.samplers[stype](240, self.phase == "train"),
            )

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            try:
                with open(self.ann_file, "r") as fin:
                    for line in fin:
                        line_split = line.strip().split(",")
                        filename, _, _, label = line_split
                        label = float(label)
                        filename = osp.join(self.data_prefix, filename)
                        self.video_infos.append(dict(filename=filename, label=label))
            except:
                #### No Label Testing
                video_filenames = []
                for root, dirs, files in os.walk(self.data_prefix, topdown=True):
                    for file in files:
                        if file.endswith(".mp4"):
                            video_filenames += [os.path.join(root, file)]
                print(len(video_filenames))
                video_filenames = sorted(video_filenames)
                for filename in video_filenames:
                    self.video_infos.append(dict(filename=filename, label=-1))

    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        label = video_info["label"]

        try:
            ## Read Original Frames
            ## Process Frames
            data, frame_inds = spatial_temporal_view_decomposition(
                filename,
                self.sample_types,
                self.samplers,
                self.phase == "train",
                self.augment and (self.phase == "train"),
            )

            for k, v in data.items():
                data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(
                    3, 0, 1, 2
                )

            data["num_clips"] = {}
            for stype, sopt in self.sample_types.items():
                data["num_clips"][stype] = sopt["num_clips"]
            data["frame_inds"] = frame_inds
            data["gt_label"] = label
            data["name"] = filename  # osp.basename(video_info["filename"])
        except:
            # exception flow
            return {"name": filename}

        return data

    def __len__(self):
        return len(self.video_infos)
