"""Observation key mapping helpers for inference payloads."""
from __future__ import annotations

from typing import Any

import numpy as np


def build_camera_key_map(
    cameras: dict[str, Any],
    enabled_cameras: list[str] | None = None,
    tactile: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build local image sensor name -> inference-key mapping from config.

    Local sensor management, display, and recording keep the YAML sensor names.
    Only inference payload image keys are remapped.
    """
    tactile = tactile or {}
    names = (
        enabled_cameras
        if enabled_cameras is not None
        else [*cameras.keys(), *tactile.keys()]
    )
    key_map: dict[str, str] = {}
    reverse: dict[str, str] = {}

    for name in names:
        cfg = cameras.get(name)
        if cfg is None:
            cfg = tactile.get(name)
        if cfg is None:
            raise ValueError(
                "enabled_cameras contains unknown image sensor keys: "
                f"{name!r}"
            )
        mapped_key = _get_mapped_key(cfg) or name
        if mapped_key in reverse:
            raise ValueError(
                "image sensor mapped_key conflict: "
                f"{name!r} and {reverse[mapped_key]!r} both map to {mapped_key!r}"
            )
        key_map[name] = mapped_key
        reverse[mapped_key] = name

    return key_map


def map_image_keys(
    images: dict[str, np.ndarray],
    camera_key_map: dict[str, str],
) -> dict[str, np.ndarray]:
    """Return images keyed by the policy-facing mapped key."""
    mapped: dict[str, np.ndarray] = {}
    reverse: dict[str, str] = {}

    for name, image in images.items():
        out_key = camera_key_map.get(name, name)
        if out_key in mapped:
            raise ValueError(
                "image sensor mapped_key conflict in image payload: "
                f"{name!r} and {reverse[out_key]!r} both map to {out_key!r}"
            )
        mapped[out_key] = image
        reverse[out_key] = name

    return mapped


def _get_mapped_key(config: Any) -> str | None:
    if config is None:
        return None
    if isinstance(config, dict):
        value = config.get("mapped_key")
    else:
        value = getattr(config, "mapped_key", None)
    if value is None:
        return None
    value = str(value).strip()
    return value or None
