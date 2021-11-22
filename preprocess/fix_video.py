import os
import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np
import skvideo.io


def fix_single_video(video_path: Union[str, Path]) -> bool:
    """
    Fix a single video file.

    :param video_path: Path to the video file.
    :return: True if the video was fixed, False otherwise.
    """

    video_path = Path(video_path)
    expected = False
    try:
        video_data: np.ndarray = skvideo.io.vread(str(video_path))
    except Exception as e:
        expected = True

    if not expected:
        print(f"OK. No need to fix video {video_path}")
        return True

    tmp_path = video_path.parent / (video_path.stem + "_tmp.mp4")
    cmd = f"ffmpeg -v quiet -i {video_path} -c:v libx264 -c:a copy {tmp_path}"
    subprocess.run(cmd, shell=True)

    video_data = None
    try:
        video_data: np.ndarray = skvideo.io.vread(tmp_path)
    except Exception as e:
        print(f"!!! Error persists. {str(video_path)} not fixed.")
        return False

    if video_data is not None:
        print(f"OK. {video_path} is fixed")
        os.remove(str(video_path))
        tmp_path.rename(video_path)

    return True


def get_failed_videos(log_file: Union[str, Path]) -> List[Path]:
    """
    Get the list of failed videos.

    :param video_dir: Path to the video directory.
    :return: List of failed videos.
    """

    log_file = Path(log_file)
    failed_videos = []

    with log_file.open() as f:
        for line in f.readlines():
            if "file error:" in line:
                failed_videos.append(Path(line.split()[-1].strip()))

    return failed_videos


def main():
    failed_videos = get_failed_videos("logs/error.log")
    fixed_count = 0
    failed_count = 0
    for i in failed_videos:
        fixed = fix_single_video(i)

    if fixed:
        fixed_count += 1
    else:
        failed_count += 1

    print(f"Fixed {fixed_count} videos.")
    print(f"Failed {failed_count} videos.")


if __name__ == "__main__":
    main()
