import shutil
import subprocess
from pathlib import Path

import numpy as np
import pycolmap
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import CameraCalibration, ImuCalibration
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from scipy.spatial.transform import Rotation

from .. import logger
from .constants import (
    LEFT_CAMERA_STREAM_ID,
    RIGHT_CAMERA_STREAM_ID,
    RIGHT_IMU_STREAM_ID,
)

# ----- Camera functions ----- #


def get_camera_params_for_colmap(
    camera_calibration: CameraCalibration,
    camera_model: str,
) -> list[float]:
    """
    Convert Aria CameraCalibration to COLMAP camera parameters.
    Supported models: OPENCV_FISHEYE, FULL_OPENCV, RAD_TAN_THIN_PRISM_FISHEYE
    Args:
        camera_calibration (CameraCalibration):
        The projectaria_tools CameraCalibration object
        camera_model (str): The COLMAP camera model to use
    Returns:
        list[float]: The camera parameters in COLMAP format
    """
    # params = [f_u {f_v} c_u c_v [k_0: k_{numK-1}]
    # {p_0 p_1} {s_0 s_1 s_2 s_3}]
    # projection_params is a 15 length vector,
    # starting with focal length, pp, extra coeffs
    camera_params = camera_calibration.get_projection_params()
    f_x, f_y, c_x, c_y = (
        camera_params[0],
        camera_params[0],
        camera_params[1],
        camera_params[2],
    )

    p2, p1 = camera_params[-5], camera_params[-6]

    k1, k2, k3, k4, k5, k6 = camera_params[3:9]

    # FULL_OPENCV model format:
    # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

    # OPENCV_FISHEYE model format:
    # fx, fy, cx, cy, k1, k2, k3, k4

    if camera_model == "OPENCV_FISHEYE":
        params = [f_x, f_y, c_x, c_y, k1, k2, k3, k4]
    elif camera_model == "FULL_OPENCV":
        params = [f_x, f_y, c_x, c_y, k1, k2, p1, p2, k3, k4, k5, k6]
    elif camera_model == "RAD_TAN_THIN_PRISM_FISHEYE":
        aria_fisheye_params = camera_params
        focal_length = aria_fisheye_params[0]
        aria_fisheye_params = np.insert(aria_fisheye_params, 0, focal_length)
        params = aria_fisheye_params

    return params


def camera_colmap_from_calib(calib: CameraCalibration) -> pycolmap.Camera:
    """Loads pycolmap camera from Aria CameraCalibration object"""
    if calib.get_model_name().name != "FISHEYE624":
        raise ValueError(
            f"Unsupported Aria model {calib.get_model_name().name}"
        )
    model = "RAD_TAN_THIN_PRISM_FISHEYE"
    params = get_camera_params_for_colmap(calib, model)
    width, height = calib.get_image_size()
    return pycolmap.Camera(
        model=model,
        width=width,
        height=height,
        params=params,
    )


# ----- Transformation functions ----- #


def get_t_imu_camera(
    imu_calib: ImuCalibration,
    camera_calib: CameraCalibration,
) -> pycolmap.Rigid3d:
    """Get T_imu_camera from Aria calibrations.
    Returns pycolmap.Rigid3d transform.
    """

    t_device_cam = camera_calib.get_transform_device_camera()
    t_device_imu = imu_calib.get_transform_device_imu()
    t_imu_device = t_device_imu.inverse()

    t_imu_cam = t_imu_device @ t_device_cam

    colmap_t_imu_cam = rigid3d_from_transform(t_imu_cam)

    return colmap_t_imu_cam


def rigid3d_from_transform(transform: SE3) -> pycolmap.Rigid3d:
    """Converts projectaria_tools Rigid3d to pycolmap Rigid3d

    Note: to_quat() returns in wxyz format, but pycolmap.Rotation3d
    expects xyzw format."""

    # https://github.com/facebookresearch/projectaria_tools/blob/867105e65cadbe777db355a407d90533c71d2d06/core/python/sophus/SO3PyBind.h#L161

    qvec = transform.rotation().to_quat()[0]
    tvec = transform.translation()[0]
    qvec = np.roll(
        qvec, -1
    )  # change from w,x,y,z to x,y,z,w for pycolmap format
    q = np.array(qvec)
    t = np.array(tvec)

    return pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)


def get_magnitude_from_transform(
    transform: pycolmap.Rigid3d,
) -> tuple[float, float]:
    """Returns rotation (in degrees) and
    translation (in meters) magnitudes
    from a Rigid3d transform
    """
    translation = transform.translation
    quat_xyzw = transform.rotation.quat
    rotation = Rotation.from_quat(quat_xyzw)
    dr = np.rad2deg(rotation.magnitude())
    dt = np.linalg.norm(translation)

    return dr, dt


# ----- VRS utils ----- #


def initialize_reconstruction_from_vrs_file(
    vrs_file: Path,
) -> pycolmap.Reconstruction:
    """
    Return a pycolmap.Reconstruction with only cameras and rigs populated.
    """
    # Initialize VRS data provider
    vrs_provider = data_provider.create_vrs_data_provider(vrs_file.as_posix())

    # build reconstruction
    recon = pycolmap.Reconstruction()
    device_calibration = vrs_provider.get_device_calibration()
    imu_stream_label = vrs_provider.get_label_from_stream_id(
        StreamId(RIGHT_IMU_STREAM_ID)
    )
    imu_calib = device_calibration.get_imu_calib(imu_stream_label)

    rig = pycolmap.Rig(rig_id=1)

    # DUMMY CAMERA FOR IMU, IMU ID is 1
    imu = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", params=[0, 0, 0])
    recon.add_camera(imu)
    rig.add_ref_sensor(imu.sensor_id)

    for cam_id, sid in [
        (2, StreamId(LEFT_CAMERA_STREAM_ID)),
        (3, StreamId(RIGHT_CAMERA_STREAM_ID)),
    ]:
        stream_label = vrs_provider.get_label_from_stream_id(sid)
        camera_calib = device_calibration.get_camera_calib(stream_label)
        cam = camera_colmap_from_calib(camera_calib)
        cam.camera_id = cam_id

        t_imu_camera = get_t_imu_camera(
            imu_calib,
            camera_calib,
        )
        sensor_from_rig = t_imu_camera.inverse()

        recon.add_camera(cam)
        rig.add_sensor(cam.sensor_id, sensor_from_rig)

    recon.add_rig(rig)
    return recon


def extract_images_from_vrs(
    vrs_file: Path,
    image_folder: Path,
    left_subfolder_name="left",
    right_subfolder_name="right",
    verbose: bool = False,
):
    for camera, stream_id in [
        (left_subfolder_name, LEFT_CAMERA_STREAM_ID),
        (right_subfolder_name, RIGHT_CAMERA_STREAM_ID),
    ]:
        output_dir = image_folder / camera
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(output_dir)

        logger.info(
            "Extracting images for camera %s in VRS %s", camera, vrs_file
        )
        cmd = f"vrs extract-images {vrs_file} --to {output_dir} + {stream_id}"
        stdout = None if verbose else subprocess.PIPE
        out = subprocess.run(
            cmd, shell=True, stderr=subprocess.STDOUT, stdout=stdout
        )
        if out.returncode:
            msg = f"Command '{cmd}' returned {out.returncode}."
            if out.stdout:
                msg += "\n" + out.stdout.decode("utf-8")
            raise subprocess.SubprocessError(msg)
        logger.info("Done!")


def _image_names_from_folder(
    folder: Path, wrt_to: Path, ext: str = ".jpg"
) -> list[Path]:
    if not folder.is_dir():
        return []
    images = sorted(n for n in folder.iterdir() if n.suffix == ext)
    images = [n.relative_to(wrt_to) for n in images]
    return images


def extract_images_with_timestamps_from_vrs(
    vrs_file, images_path
) -> dict[int, tuple[Path, Path]]:
    """
    Return timestamps -> image names
    """

    # Initialize VRS data provider
    vrs_provider = data_provider.create_vrs_data_provider(vrs_file.as_posix())
    left_ts = sorted(
        vrs_provider.get_timestamps_ns(
            StreamId(LEFT_CAMERA_STREAM_ID), TimeDomain.DEVICE_TIME
        )
    )

    # Extract images from VRS file
    extract_images_from_vrs(
        vrs_file=vrs_file,
        image_folder=images_path,
    )

    # Get images
    left_img_dir = images_path / "left"
    right_img_dir = images_path / "right"
    left_images = _image_names_from_folder(left_img_dir, left_img_dir)
    right_images = _image_names_from_folder(right_img_dir, right_img_dir)
    images = list(zip(left_images, right_images))

    # Create a map
    assert len(left_ts) == len(images), (
        "timestamps should have the same length as images"
    )
    return dict(zip(left_ts, images))
