from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pycolmap

from .. import logger


def _round_ns(x: str | int | float) -> int:
    # works for "5120...274.999939", "5.12e11", or ints
    if isinstance(x, int):
        return x
    s = str(x)
    return int(Decimal(s).to_integral_value(rounding=ROUND_HALF_UP))


@dataclass(slots=True)
class Trajectory:
    """
    Loads and stores traj data from 'estimate' text file with rows:
      ts t_x t_y t_z q_x q_y q_z q_w
    Blank lines and lines starting with '#' are ignored.

    By default, poses are calculated as rig_from_world
    (i.e., inverse of world_from_rig) to satisfy COLMAP format.

    Attributes:
        invert_poses (bool): Whether to invert poses to
        rig_from_world format.
        corresponding_sensor (str): The reference sensor to in which
        the trajectory is represented ("imu" or "cam0").
    """

    invert_poses: bool = True
    corresponding_sensor: str = "imu"
    _timestamps: list[int] = field(default_factory=list[int])
    _poses: list[pycolmap.Rigid3d] = field(
        default_factory=list[pycolmap.Rigid3d]
    )

    @classmethod
    def load_from_file(
        cls,
        path: str | Path,
        invert_poses: bool = True,
        corresponding_sensor: str = "imu",
    ) -> "Trajectory":
        """Parse the file, validate format, populate timestamps & poses."""
        self = cls()
        self.clear()
        path = Path(path)
        self.invert_poses = invert_poses
        self.corresponding_sensor = corresponding_sensor

        if not path.exists():
            raise FileNotFoundError(f"Estimate file not found: {path}")

        with open(path) as f:
            lines = f.readlines()

        state = self._parse(lines)
        if not state:
            raise RuntimeError("Failed to parse estimate file.")

        return self

    @property
    def timestamps(self) -> list[int]:
        self._ensure_loaded()
        return self._timestamps

    @property
    def poses(self) -> list[pycolmap.Rigid3d]:
        self._ensure_loaded()
        return self._poses

    @property
    def positions(self) -> np.ndarray:
        """Returns Nx3 numpy array of positions."""
        self._ensure_loaded()
        if not self.invert_poses:
            # poses are in world_from_rig format
            return np.array([p.translation for p in self._poses])
        else:
            return np.array([p.inverse().translation for p in self._poses])

    @property
    def orientations(self) -> np.ndarray:
        """Returns Nx4 numpy array of quaternions (x, y, z, w)."""
        self._ensure_loaded()
        if not self.invert_poses:
            # poses are in world_from_rig format
            return np.array([p.rotation.quat for p in self._poses])
        else:
            # poses are in rig_from_world format
            return np.array([p.inverse().rotation.quat for p in self._poses])

    def as_tuples(self) -> list[tuple[int, pycolmap.Rigid3d]]:
        """Return a list of (timestamp, pose) tuples."""
        self._ensure_loaded()
        return list(zip(self._timestamps, self._poses, strict=True))

    def as_dict(self) -> dict[int, pycolmap.Rigid3d]:
        """Return a dict mapping timestamp to pose."""
        self._ensure_loaded()
        return dict(zip(self._timestamps, self._poses, strict=True))

    def __len__(self) -> int:
        return len(self._timestamps)

    def is_loaded(self) -> bool:
        return len(self._timestamps) > 0

    def clear(self) -> None:
        self._timestamps.clear()
        self._poses.clear()

    def _ensure_loaded(self) -> None:
        if not self._timestamps or not self._poses:
            raise RuntimeError(
                "Estimate not loaded. Call load_from_file() first."
            )

    # Parses the file lines, populating self._timestamps and self._poses
    def _parse(self, lines: list[str]) -> None:
        ts_list: list[int] = []
        pose_list: list[pycolmap.Rigid3d] = []
        exists_lines = False
        for lineno, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            exists_lines = True
            parts = line.split()
            if len(parts) != 8:
                logger.error(
                    f"Line {lineno}: expected 8 values, got {len(parts)}"
                )
                return False

            try:
                ts = _round_ns(parts[0])
                tvec = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
                qvec = np.array(
                    [
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    ]
                )

            except ValueError as e:
                logger.error(f"Line {lineno}: invalid number format: {e}")
                return False

            world_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec), tvec)
            pose = (
                world_from_rig.inverse()
                if self.invert_poses
                else world_from_rig
            )

            ts_list.append(ts)
            pose_list.append(pose)

        if not exists_lines:
            logger.error("No valid lines found in the estimate file.")
            return False

        self._timestamps = ts_list
        self._poses = pose_list

        return True
