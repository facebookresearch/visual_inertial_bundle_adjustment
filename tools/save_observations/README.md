# LaMAria (Stripped-down Version)

This is a **stripped-down version** of the [LaMAria repository](https://github.com/cvg/lamaria) from CVG (Computer Vision Group, ETH Zurich).

For the full dataset, documentation, and benchmark details, visit the [LaMAria project page](https://lamaria.ethz.ch).

## Purpose

This repository contains the **minimum required code** to process VRS files containing data from Aria glasses. It enables generating:

- `session_observations.csv` — image point observations used for visual-inertial bundle adjustment
- `vrs_source_info.json` — description of camera/IMU layout

This stripped-down code is only meant to be a tool for the [visual_inertial_bundle_adjustment](https://github.com/facebookresearch/visual_inertial_bundle_adjustment) project to generate the required input data.

## Installation

Install dependencies (using a virtual environment is recommended):

```bash
pip install -r requirements.txt
```

Additionally, you need to install the [VRS CLI tool](https://facebookresearch.github.io/vrs/docs/VrsCliTool/) to extract images from VRS files.

## Usage

Run from the project root directory:

```bash
python -m save_observations --output $OUTPUT \
       --vrs $VRS \
       --mps-path $MPS_DATA
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--output` | Output directory for generated files |
| `--vrs` | Path to the input VRS file from Aria glasses |
| `--mps-path` | Path to folder containing MPS (Machine Perception Services) data |
| `--trajectory-type` | Trajectory type: `closed_loop` (default) or `open_loop` |

### MPS Data Requirements

The `$MPS_DATA` folder must contain:

- `online_calibration.jsonl`
- **Either** `closed_loop_framerate_trajectory.csv` (default) **or** `open_loop_trajectory.csv`

To use open-loop trajectory instead of closed-loop:

```bash
python -m save_observations --output $OUTPUT \
       --vrs $VRS \
       --mps-path $MPS_DATA \
       --trajectory-type open_loop
```

## License

See [LICENSE](LICENSE) (MIT).
