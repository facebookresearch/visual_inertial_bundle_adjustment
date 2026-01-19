# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import pandas as pd
import pycolmap
from tqdm import tqdm

from lamaria.config.options import (
    KeyframeSelectorOptions,
    TriangulatorOptions,
)
from lamaria.config.pipeline import PipelineOptions
from lamaria.pipeline.estimate_to_timed_reconstruction import (
    convert_estimate_into_timed_reconstruction,
)
from lamaria.pipeline.keyframe_selection import KeyframeSelector
from lamaria.pipeline.triangulation import run as triangulate
from lamaria.structs.timed_reconstruction import TimedReconstruction
from lamaria.structs.trajectory import Trajectory
from lamaria.utils.aria import (
    extract_images_with_timestamps_from_vrs,
    initialize_reconstruction_from_vrs_file,
)


# (from example_vi_optimization.py)
def run_estimate_to_timed_recon(
    vrs: Path,
    images_path: Path,
    estimate: Path,
) -> TimedReconstruction:
    """Function to convert a general input
    estimate file to a TimedReconstruction.
    """
    traj = Trajectory.load_from_file(estimate)
    init_recon = initialize_reconstruction_from_vrs_file(vrs)
    timestamps_to_images = extract_images_with_timestamps_from_vrs(
        vrs, images_path
    )
    timed_recon = convert_estimate_into_timed_reconstruction(
        init_recon, traj, timestamps_to_images
    )
    return timed_recon


# (from example_vi_optimization.py)
def run_keyframe_selection(
    options: KeyframeSelectorOptions,
    input_recon: TimedReconstruction,
    images_path: Path,
    keyframes_path: Path,
) -> TimedReconstruction:
    kf_vi_recon = KeyframeSelector.run(
        options,
        input_recon,
        images_path,
        keyframes_path,
    )
    return kf_vi_recon


# (from example_vi_optimization.py)
def run_triangulation(
    options: TriangulatorOptions,
    reference_model_path: Path,
    keyframes_path: Path,
    triangulation_path: Path,
) -> pycolmap.Reconstruction:
    triangulated_model_path = triangulate(
        options,
        reference_model_path,
        keyframes_path,
        triangulation_path,
    )
    return pycolmap.Reconstruction(triangulated_model_path)


RECON_CAMERA_ID_TO_LABEL = {
    2: "camera-slam-left",
    3: "camera-slam-right",
}

CAMERAS_LAYOUT = [
    "camera-slam-left",
    "camera-slam-right",
]

CAMERA_LABEL_TO_INDEX = {camera: i for i, camera in enumerate(CAMERAS_LAYOUT)}

CSV_FIELDS = [
    "point_id",
    "capture_timestamp_ns",
    "camera_index",
    "projection_base_res_x",
    "projection_base_res_y",
    "sqrt_h_base_res_00",
    "sqrt_h_base_res_01",
    "sqrt_h_base_res_10",
    "sqrt_h_base_res_11",
]

# TODO: obtain proper image dector corner's standard deviation
DEFAULT_SQRT_H_BASE_RES = [0.7, 0.0, 0.0, 0.7]


def save_recon_observations_to_csv(recon: TimedReconstruction, csv_path: Path):
    rows = []

    cam_stats = {
        label: {
            "min_x": float("inf"),
            "min_y": float("inf"),
            "max_x": float("-inf"),
            "max_y": float("-inf"),
        }
        for label in CAMERAS_LAYOUT
    }

    # iterate over frame (with its multiple images)
    for f_id, frame in tqdm(recon.reconstruction.frames.items()):
        capture_timestamp_ns = recon.timestamps.get(f_id)
        assert capture_timestamp_ns is not None
        capture_timestamp_us = capture_timestamp_ns // 1000

        for data in frame.data_ids:
            image_id = data.id
            camera_id = data.sensor_id.id
            camera_label = RECON_CAMERA_ID_TO_LABEL.get(camera_id)
            assert camera_label is not None

            camera_index = CAMERA_LABEL_TO_INDEX.get(camera_label)
            assert camera_index is not None

            image = recon.reconstruction.images[image_id]

            this_cam_stats = cam_stats[camera_label]
            for obs in image.get_observation_points2D():
                this_cam_stats["min_x"] = min(
                    float(obs.xy[0]), this_cam_stats["min_x"]
                )
                this_cam_stats["max_x"] = max(
                    float(obs.xy[0]), this_cam_stats["max_x"]
                )
                this_cam_stats["min_y"] = min(
                    float(obs.xy[1]), this_cam_stats["min_y"]
                )
                this_cam_stats["max_y"] = max(
                    float(obs.xy[1]), this_cam_stats["max_y"]
                )
                rows.append(
                    [
                        obs.point3D_id,
                        capture_timestamp_us,
                        camera_index,
                        float(obs.xy[0]),
                        float(obs.xy[1]),
                        *DEFAULT_SQRT_H_BASE_RES,
                    ]
                )
    print(f"Observation stats: {cam_stats}")

    print("Saving to CSV...")
    df = pd.DataFrame(data=rows, columns=CSV_FIELDS)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


def save_sensor_layout(mps_online_calibration: Path, output: Path):
    with open(mps_online_calibration) as f:
        calib = json.loads(f.readline())
    cam_label_to_serial = {
        c["Label"]: c["SerialNumber"] for c in calib["CameraCalibrations"]
    }

    layout = {
        "camera_ids": [
            cam_label_to_serial.get(label) for label in CAMERAS_LAYOUT
        ],
        "imu_ids": ["imu-right", "imu-left"],
    }

    with open(output, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Sensor layout saved to: {output}")

    # compute T_device_imu0
    calib_imu0 = next(
        imu_calib
        for imu_calib in calib["ImuCalibrations"]
        if imu_calib["Label"] == "imu-right"
    )
    tvec = calib_imu0["T_Device_Imu"]["Translation"]
    q_w, q_xyz = calib_imu0["T_Device_Imu"]["UnitQuaternion"]
    T_device_imu0 = pycolmap.Rigid3d(pycolmap.Rotation3d([*q_xyz, q_w]), tvec)

    return T_device_imu0


# Convert to:
#   timestamp tx ty tz qx qy qz qw # <- T_world_rig, where rig = imu0
# (https://github.com/cvg/lamaria?tab=readme-ov-file#input-format)
def convert_closed_loop_trajectory(
    mps_closed_loop_trajectory: Path,
    T_device_imu0: pycolmap.Rigid3d,
    estimate: Path,
):
    df = pd.read_csv(mps_closed_loop_trajectory)

    rows = []
    for _, r in df.iterrows():
        timestamp_ns = r["tracking_timestamp_us"] * 1000
        T_world_device = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                [
                    r["qx_world_device"],
                    r["qy_world_device"],
                    r["qz_world_device"],
                    r["qw_world_device"],
                ]
            ),
            [
                r["tx_world_device"],
                r["ty_world_device"],
                r["tz_world_device"],
            ],
        )
        T_world_imu0 = T_world_device * T_device_imu0
        row = [
            timestamp_ns,
            *T_world_imu0.translation,
            *T_world_imu0.rotation.quat,
        ]
        rows.append(row)

    print(f"writing {len(rows)} poses converted from CLOSED loop trajectory")
    cvg_traj = pd.DataFrame(
        rows, columns=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    )
    cvg_traj.to_csv(estimate, sep=" ", header=False, index=False)


# as above, but from MPS open loop (take only timestamps in online calibration)
def convert_open_loop_trajectory(
    mps_open_loop_trajectory: Path,
    mps_online_calibration: Path,  # for time stamps
    T_device_imu0: pycolmap.Rigid3d,
    estimate: Path,
):
    with open(mps_online_calibration) as f:
        frame_timestamps = {
            json.loads(line)["tracking_timestamp_us"] for line in f
        }

    df = pd.read_csv(mps_open_loop_trajectory)

    rows = []
    for _, r in df.iterrows():
        timestamp_us = r["tracking_timestamp_us"]
        if timestamp_us not in frame_timestamps:
            continue

        timestamp_ns = timestamp_us * 1000
        T_world_device = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                [
                    r["qx_odometry_device"],
                    r["qy_odometry_device"],
                    r["qz_odometry_device"],
                    r["qw_odometry_device"],
                ]
            ),
            [
                r["tx_odometry_device"],
                r["ty_odometry_device"],
                r["tz_odometry_device"],
            ],
        )
        T_world_imu0 = T_world_device * T_device_imu0
        row = [
            timestamp_ns,
            *T_world_imu0.translation,
            *T_world_imu0.rotation.quat,
        ]
        rows.append(row)

    print(f"writing {len(rows)} poses converted from OPEN loop trajectory")
    cvg_traj = pd.DataFrame(
        rows, columns=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    )
    cvg_traj.to_csv(estimate, sep=" ", header=False, index=False)


def run_pipeline(
    options: PipelineOptions,
    vrs: Path,
    output_path: Path,
    mps_path: Path,
    trajectory_type: str,
):
    mps_online_calibration = mps_path / "online_calibration.jsonl"

    # save sensor layout
    layout_output = output_path / "vrs_source_info.json"
    T_device_imu0 = save_sensor_layout(mps_online_calibration, layout_output)

    # convert to CVG format, to run pipeline
    estimate = output_path / "estimated_trajectory.txt"
    if trajectory_type == "closed_loop":
        mps_closed_loop_trajectory = (
            mps_path / "closed_loop_framerate_trajectory.csv"
        )
        convert_closed_loop_trajectory(
            mps_closed_loop_trajectory, T_device_imu0, estimate
        )
    else:
        mps_open_loop_trajectory = mps_path / "open_loop_trajectory.csv"
        convert_open_loop_trajectory(
            mps_open_loop_trajectory,
            mps_online_calibration,
            T_device_imu0,
            estimate,
        )

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    recon = None

    # Estimate to Lamaria Reconstruction
    image_path = output_path / "images"
    init_recon_path = output_path / "initial_recon"
    if init_recon_path.exists():
        recon = TimedReconstruction.read(init_recon_path)
    else:
        recon = run_estimate_to_timed_recon(
            vrs,
            output_path / "images",
            estimate,
        )

    # Keyframe Selection
    keyframe_path = output_path / "keyframes"
    keyframed_recon_path = output_path / "keyframed_recon"
    if keyframed_recon_path.exists():
        recon = TimedReconstruction.read(keyframed_recon_path)
    else:
        recon = run_keyframe_selection(
            options.keyframing_options,
            recon,
            image_path,
            keyframe_path,
        )
        recon.write(output_path / "keyframed_recon")

    # Triangulation
    triangulation_path = output_path / "triangulated"
    tri_model_path = triangulation_path / "model"
    if tri_model_path.exists():
        recon = TimedReconstruction.read(tri_model_path)
    else:
        pycolmap_recon = run_triangulation(
            options.triangulator_options,
            keyframed_recon_path,
            keyframe_path,
            triangulation_path,
        )
        recon = TimedReconstruction(
            reconstruction=pycolmap_recon, timestamps=recon.timestamps
        )
        recon.write(tri_model_path)

    # Save observations to CSV
    csv_observations_path = output_path / "session_observations.csv"
    save_recon_observations_to_csv(recon, csv_observations_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save observations and .")
    parser.add_argument(
        "--config",
        type=str,
        default="./defaults.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--vrs",
        type=str,
        required=True,
        help="Path to the input VRS file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--mps-path",
        type=str,
        required=True,
        help="Path with MPS estimates (online_calibration.jsonl, and either"
        + " closed_loop_framerate_trajectory.csv|open_loop_trajectory.csv)",
    )
    parser.add_argument(
        "--trajectory-type",
        type=str,
        default="closed_loop",
        help="The trajectory to be loaded to find observations"
        + " (closed_loop|open_loop)",
    )
    args = parser.parse_args()

    assert args.trajectory_type in {"closed_loop", "open_loop"}

    options = PipelineOptions()
    options.load(args.config)
    run_pipeline(
        options,
        Path(args.vrs),
        Path(args.output),
        Path(args.mps_path),
        args.trajectory_type,
    )
