from bisect import bisect_left
from copy import deepcopy
from pathlib import Path

import pycolmap

from ..structs.timed_reconstruction import TimedReconstruction
from ..structs.trajectory import (
    Trajectory,
)


def _image_names_from_folder(
    folder: Path, wrt_to: Path, ext: str = ".jpg"
) -> list[Path]:
    if not folder.is_dir():
        return []
    images = sorted(n for n in folder.iterdir() if n.suffix == ext)
    images = [n.relative_to(wrt_to) for n in images]
    return images


def _match_estimate_ts_to_images(
    timestamps_to_images: dict[int, tuple[Path, Path]],
    estimate: Trajectory,
    max_diff: int = 1000000,  # 1 ms
) -> tuple[list, list]: # Return matched Poses and matched Timestamps
    left_ts = sorted(timestamps_to_images.keys())
    
    matched_poses = []
    matched_img_timestamps = []

    # Iterate over the images as the anchor
    for img_ts in left_ts:
        # Find the closest timestamp in the trajectory
        idx = bisect_left(estimate.timestamps, img_ts)

        cand_idxs = []
        if idx > 0:
            cand_idxs.append(idx - 1)
        if idx < len(estimate.timestamps):
            cand_idxs.append(idx)

        if not cand_idxs:
            continue

        best_idx = min(cand_idxs, key=lambda j: abs(estimate.timestamps[j] - img_ts))
        
        # Check if the trajectory has a pose close enough to this image
        if (max_diff is not None) and (abs(estimate.timestamps[best_idx] - img_ts) > max_diff):
            continue

        matched_poses.append(estimate.poses[best_idx])
        matched_img_timestamps.append(img_ts)
        
    return matched_poses, matched_img_timestamps


def convert_estimate_into_timed_reconstruction(
    init_reconstruction: pycolmap.Reconstruction,
    estimate: Trajectory,
    timestamps_to_images: dict[int, tuple[Path, Path]],
) -> TimedReconstruction:
    """
    Populate a TimedReconstruction from a trajectory
    """
    
    # Filter the trajectory down to only the poses that match an image
    matched_poses, matched_timestamps = _match_estimate_ts_to_images(
        timestamps_to_images, estimate
    )

    assert len(matched_poses) == len(matched_timestamps), "Mismatch in matched pairs"

    recon = deepcopy(init_reconstruction)
    image_id = 1
    frame_id_to_timestamp = dict()
    for frame_id, (pose, timestamp) in enumerate(
        zip(matched_poses, matched_timestamps)
    ):
        frame = pycolmap.Frame()
        frame.rig_id = 1
        frame.frame_id = frame_id
        frame.rig_from_world = pose  # as it corresponds to imu

        image_names = timestamps_to_images[timestamp]
        images_to_add = []
        for cam_id, img_path in [(2, image_names[0]), (3, image_names[1])]:
            im = pycolmap.Image(
                str(img_path),
                pycolmap.Point2DList(),
                cam_id,
                image_id,
            )
            im.frame_id = frame.frame_id
            frame.add_data_id(im.data_id)
            images_to_add.append(im)
            image_id += 1
        recon.add_frame(frame)
        for im in images_to_add:
            recon.add_image(im)
        frame_id_to_timestamp[frame_id] = timestamp

    return TimedReconstruction(
        reconstruction=recon, timestamps=frame_id_to_timestamp
    )
