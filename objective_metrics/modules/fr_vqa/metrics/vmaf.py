import os
import re
import numpy as np
import logging
import subprocess
from typing import Literal
from ...utils.video_readers import get_metadata

VMAF_N_THREADS = int(os.environ.get("VMAF_N_THREADS", 1))

class VMAF:
    def __init__(self, logger: logging.Logger):
        self.metric = "vmaf"
        self.logger = logger

    def score_video(
            self,
            vid_ref: str,
            vid_dist: str, 
            mode: Literal["same", "repeat_each", "repeat_last"],
    ) -> float:
        """Compute VMAF for one video. Using FFMPEG libvmaf via shell command.

        Parameters
        ----------
        vid_ref : str
            path to reference video
        vid_dist : str
            path to distorted video
        mode : Literal["same", "repeat_each", "repeat_last"]
            how to treat videos with different length

        Returns
        -------
        float
            VMAF score
        """

        if mode not in ["same", "repeat_each", "repeat_last"]:
            self.logger.error(f"Unknown mode={mode}. Provide one of: same, repeat_each, repeat_last")
            return np.nan
        
        if not os.path.isfile(vid_ref):
            self.logger.error(f"Ref video: {vid_ref} doesn't exist!")
            return np.nan

        if not os.path.isfile(vid_dist):
            self.logger.error(f"Dist video: {vid_dist} doesn't exist!")
            return np.nan

        fps_ref, height_ref, width_ref = get_metadata(vid_ref)
        fps_dist, height_dist, width_dist = get_metadata(vid_dist)
        
        # <----------------------- Make Filters ----------------------->
        setpts_filter = "setpts=PTS-STARTPTS"

        dist_filters = []
        vmaf_options = [f"libvmaf=n_threads={VMAF_N_THREADS}"]

        if mode in ["same", "repeat_each"]:
            vmaf_options.append("shortest=1")
        elif mode == "repeat_last":
            vmaf_options.append("repeatlast=1")

        if fps_dist != fps_ref:
            dist_filters.append(f"framerate={fps_ref}")

        if height_dist != height_ref or width_ref != width_dist:
            dist_filters.append(f"scale={width_ref}:{height_ref}:flags=bicubic,setsar=1")
        
        dist_filters.append(setpts_filter)
        # <------------------------------------------------------------>

        ref_chain = "[0:v]" + setpts_filter + "[reference];"
        dist_chain = "[1:v]" + ",".join(dist_filters) + "[distorted];"
        vmaf_chain = "[distorted][reference]" + ":".join(vmaf_options)

        cmd = (
            "ffmpeg"
            + " -i " + '"' + vid_ref + '"'
            + " -i " + '"' + vid_dist + '"'
            + " -lavfi " + '"' + ref_chain + dist_chain + vmaf_chain + '"'
            + " -f " + "null -"
        )
        
        self.logger.info(f"Executing ffmpeg command: {cmd}")
        out = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        self.logger.debug(f"FFMPEG output: {out}")

        score = re.compile(r"VMAF score:\s*([-+]?\d*\.?\d+|[-+]?\d+)")

        return float(score.search(out).group(1))
