[![CI](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/blob/main/LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![PRs](https://img.shields.io/badge/PRs-welcome-green.svg)](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/blob/main/CONTRIBUTING.md)

# Visual Inertial Bundle Adjustment

This is an implementation of a full sensor-modelling visual intertial bundle adjustment, designed to provide the
highest level of optimization of device states for Aria Gen1/Gen2 glasses. As a tool, it allows to reoptimize
estimates on recordings data. Since it works with a very dense problem formulation, you can expect to be able
to optimize up to 20m-30m of Aria recording.

![](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/raw/refs/heads/main/docs/img/gui_animation.gif)

## How to run

Follow instrution below for building. Assuming the folder $VIBA_INPUT_DIR has input data, you can run:

```
build/interfaces/ark/ark_vi_ba -i $VIBA_INPUT_DIR -o $OUTPUT_DIR
```

A visualization of the optimization process can also be displayed on selected platforms (Linux-X11/Mac at this stage):

```
build/interfaces/ark/ark_vi_ba -i $VIBA_INPUT_DIR -o $OUTPUT_DIR
```

The input data in the folder $VIBA_INPUT_DIR should be:

```
factory_calibration.json   # factory calibration, as extracted from VRS file
imu_samples_imu-left.csv   # IMU measurements (left-IMU), as extracted from VRS file
imu_samples_imu-right.csv  # IMU measurements (right-IMU), as extracted from VRS file
online_calibration.jsonl   # online calibration (obtained from MPS)
open_loop_trajectory.csv   # open-loop trajectory (obtained from MPS)
session_observations.csv   # recording observation (tracking points + SLAM loop-closing points)
vrs_source_info.json       # description of camera/imu layout during SLAM (indices in session_observations.csv)
```

While outputs in $OUTPUT_DIR are:

```
closed_loop_framerate_trajectory.csv  # re-estimated trajectory
online_calibration.jsonl              # re-estimated calibration
open_loop_framerate_trajectory.csv    # re-estimated trajectory (same data, in different format)
```

Notice that the files `closed_loop_framerate_trajectory.csv` and `open_loop_framerate_trajectory.csv` contain
the same data, just in different file format for convenience.

### From MPS data

All the data which are not coming from MPS can be recovered from VRS file and running Colmap pipeline.

* The inputs `factory_calibration.json` and `imu_samples_*.csv` can be extracted from a VRS file running

```
build/interfaces/ark/process_vrs -i $VRS -o $VIBA_INPUT_DIR
```

* The `session_observations.csv` and `vrs_source_info.json` can be obtained using the stripped-down Python library
  included in `tools/save_observations`. This library is derived from the [LaMAria project](https://github.com/cvg/lamaria)
  (CVG, ETH Zurich) and contains the minimum required code to process VRS files and generate observations for
  visual-inertial bundle adjustment.

  To use it, first install the dependencies (from the `tools/save_observations` directory):

```
pip install -r requirements.txt
```

  Additionally, you need to install the [VRS CLI tool](https://github.com/facebookresearch/vrs) and ensure the `vrs`
  command is available in your PATH.

  Then run (from `tools/save_observations`, assuming MPS's results are in `$MPS_DATA`):

```
python -m save_observations --output $VIBA_INPUT_DIR \
       --vrs $VRS \
       --mps-path $MPS_DATA
```

## How to build

Make sure the git submodules in `deps/baspacho` and `deps/projectaria_tools` are populated. If not:

```
git submodule update --recursive
```

### Using Pixi package manager (Linux/Mac)

This is the recommended approach on Mac, and also works on Linux:

* Install pixi: <https://pixi.sh/latest/installation/>
Configure and build:

```
pixi run prepare
pixi run build
```

### Directly building with CMake (on Linux)

Install system deps, on Fedora:

```
dnf install opus-devel fmt-devel boost-devel lz4-devel \
    xxhash-devel turbojpeg-devel openblas-devel gtest-devel
```

Install system deps, on Ubuntu:

```
sudo apt-get install -y \
  libopus-dev libfmt-dev libboost-all-dev liblz4-dev libxxhash-dev \
  libturbojpeg0-dev libopenblas-dev libgtest-dev cmake build-essential
```

Configure with CMake:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

Build:

```
cmake --build build -- -j16
```

(Optional) Run Tests:

```
ctest --test-dir build
```

## Development

Install deps to enable pre-commit git hooks and format/lint:

```
dnf install clang-tools-extra python3-pip
pip install --user pre-commit
```

## Description of the model

The whole calibration state for all sensors (all intrinsics and extrinsics) is optimized, and all recording frames
are used with all visual observations.

* Inertial device poses state is formed by world pose, linear velocity and angular velocity.
* IMU calibration is fully supported (biases/scaling/non-orthogonality, plus all time offsets:
  IMU relative to tracking clock, and accel-to-gyro time offset)
* RGB camera is uses a rolling shutter aware modelling (camera time offset and readout time are optional
  variables of the optimization problem)
* Calibration's variation in time is modelled as a random walk, with windows of 5 seconds where it's assumed to be
  constant.
* All calibration state (intrinsics/extrinsics) have (weak) factory calibration priors which prevent the optimizer from
  setting them to unreasonable estimates in the portions of recordings where they are not well constrained by observations
  (for instance, in static recording portions).
* Multiple IMUs are supported, and the inertial poses are related to the estimate of instantaneous angular velocity
  (omega priors). Since the secondary IMU's extrinsics are optimization variables, this is required to prevent the
  unconstrained angular velocity from absorbing the secondary IMU's pose/velocity errors.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
