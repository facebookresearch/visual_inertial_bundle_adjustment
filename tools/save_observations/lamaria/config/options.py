from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf, open_dict


def _structured_merge_to_obj(cls, section) -> object:
    """
    Merge a YAML section onto a structured
    config made from the dataclass `cls`,
    then return a dataclass instance.
    """
    base = OmegaConf.structured(cls)
    merged = OmegaConf.merge(base, section or {})
    return OmegaConf.to_object(merged)


# Keyframing options
@dataclass(slots=True)
class KeyframeSelectorOptions:
    max_rotation: float = 20.0  # degrees
    max_distance: float = 1.0  # meters
    max_elapsed: int = int(1e9)  # 1 second in ns

    @classmethod
    def load(cls, cfg: OmegaConf | None = None) -> KeyframeSelectorOptions:
        if cfg is None:
            return cls()

        cfg = OmegaConf.create(cfg)
        with open_dict(cfg):
            if "max_elapsed" in cfg and isinstance(cfg.max_elapsed, float):
                cfg.max_elapsed = int(cfg.max_elapsed)

        obj: KeyframeSelectorOptions = _structured_merge_to_obj(cls, cfg)
        return obj


# Triangulation options
@dataclass(slots=True)
class TriangulatorOptions:
    feature_conf: str = "aliked-n16"
    matcher_conf: str = "aliked+lightglue"
    retrieval_conf: str = "netvlad"
    num_retrieval_matches: int = 5

    # colmap defaults
    merge_max_reproj_error: float = 4.0
    complete_max_reproj_error: float = 4.0
    min_angle: float = 1.5

    filter_max_reproj_error: float = 4.0
    filter_min_tri_angle: float = 1.5

    @classmethod
    def load(cls, cfg: OmegaConf | None = None) -> TriangulatorOptions:
        if cfg is None:
            return cls()

        return _structured_merge_to_obj(cls, cfg)
