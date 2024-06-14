"""This files containes functions for extracting important info from logs.
"""
import re

def parse_error_frames_fr_iqa_videoset(filename: str) -> None:
    """Extract information about failed frames during FR IQA calculations on videoset.

    This function takes log filename as parameter and writes to a {filename}_fr-iqa-corrupted-frames.txt.
    This file will conatin list of metrics, that had errors during calculation process.

    For each metric will be written all reference videos and corresponding distorted videos with
    numbers of frames, which could not be calculated.

    Parameters
    ----------
    filename : str
        path to a log file.
    """
    f = open(filename, "r")
    lines = f.readlines()

    pattern = re.compile(r"Error for (.*) on (.*) frame\. Ref: (.*), Dist: (.*)")

    log_results = {}
    for line in lines:
        search_res = pattern.search(line)
        if search_res:

            metric = search_res.group(1)
            frame_number = int(search_res.group(2))
            ref_vid = search_res.group(3)
            dist_vid = search_res.group(4)

            if log_results.get(metric) is None:
                log_results[metric] = {}

            if log_results[metric].get(ref_vid) is None:
                log_results[metric][ref_vid] = {}

            if log_results[metric][ref_vid].get(dist_vid) is None:
                log_results[metric][ref_vid][dist_vid] = []

            log_results[metric][ref_vid][dist_vid].append(frame_number)

    with open(f'{filename}_corrupted-frames.txt', 'w') as f:
        for metric, error_sets in log_results.items():
            print(file=f)
            print("-" * 300, file=f)
            print("Metric: ", metric, file=f)
            print("Number of corrupted sets: ", len(error_sets), "\n", file=f)

            for ref_name, error_set in error_sets.items():
                print("\nReference: ", ref_name, file=f)
                print("Number of corrupted videos: ", len(error_set), "\n", file=f)

                for dist_name, frames in error_set.items():
                    print("Distorted: ", dist_name, ": ", frames, file=f)

            print("-" * 300, file=f)
