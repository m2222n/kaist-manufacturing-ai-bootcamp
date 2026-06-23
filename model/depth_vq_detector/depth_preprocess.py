from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


VALID_INPUT_MODES = {"z", "zv", "xyzv", "xyznv"}


def input_channels_for_mode(mode: str) -> int:
    if mode == "z":
        return 1
    if mode == "zv":
        return 2
    if mode == "xyzv":
        return 4
    if mode == "xyznv":
        return 7
    raise ValueError(f"Unknown input mode: {mode}. Choose one of {sorted(VALID_INPUT_MODES)}")


def load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_depth(path: str | Path, depth_scale: float | None = None) -> np.ndarray:
    """Load a depth map as float32.

    Supported production formats:
      - .npy: float depth, usually meters.
      - .npz: key "depth" is used when available.
      - uint16 PNG: integer depth. Use depth_scale=0.001 for millimeters.

    The provided dataset stores metric depth in .npz and uses NaN as background.
    NaNs are intentionally preserved here; they are converted to zeros only in
    make_depth_input after a valid-depth mask has been generated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        key = "depth" if "depth" in data.files else data.files[0]
        depth = data[key].astype(np.float32)
    else:
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            # This is acceptable for quick debugging only. Do not train from 8-bit visualization PNGs.
            arr = arr[..., 0]
        depth = arr.astype(np.float32)
    if depth_scale is not None:
        depth = depth * float(depth_scale)
    return depth.astype(np.float32)


def load_id_map(path: str | Path) -> np.ndarray:
    """Load an integer ID mask from PNG-like files.

    If the file is RGB, unique colors are converted into deterministic integer IDs.
    The new dataset does not need this for scene npz files because inst_id and
    category_id are already int32 arrays inside the npz.
    """
    path = Path(path)
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].reshape(-1, 3)
        unique, inverse = np.unique(rgb, axis=0, return_inverse=True)
        ids = inverse.reshape(arr.shape[:2]).astype(np.int64)
        black_rows = np.where(np.all(unique == 0, axis=1))[0]
        if len(black_rows) > 0 and black_rows[0] != 0:
            black_old = int(black_rows[0])
            remap = np.arange(len(unique), dtype=np.int64)
            remap[0], remap[black_old] = remap[black_old], remap[0]
            inv_remap = np.zeros_like(remap)
            inv_remap[remap] = np.arange(len(remap))
            ids = inv_remap[ids]
        return ids.astype(np.int64)
    raise ValueError(f"Unsupported mask shape {arr.shape} for {path}")


def resize_depth_nan_safe(depth: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Resize a NaN-background depth image without spreading NaNs.

    Bilinear interpolation over raw NaNs makes large invalid halos. This function
    interpolates depth*valid and valid separately, then restores NaN where there
    is not enough valid support.
    """
    if size is None:
        return depth.astype(np.float32)
    h, w = size
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    depth_filled = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d_t = torch.from_numpy((depth_filled * valid)[None, None])
    v_t = torch.from_numpy(valid[None, None])
    d_r = F.interpolate(d_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    v_r = F.interpolate(v_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.empty((h, w), dtype=np.float32)
    good = v_r > 0.25
    out[good] = d_r[good] / np.maximum(v_r[good], 1e-6)
    out[~good] = np.nan
    return out.astype(np.float32)


def resize_id_map(id_map: np.ndarray | None, size: tuple[int, int] | None) -> np.ndarray | None:
    if id_map is None or size is None:
        return None if id_map is None else id_map.astype(np.int64)
    h, w = size
    t = torch.from_numpy(id_map[None, None].astype(np.float32))
    return F.interpolate(t, size=(h, w), mode="nearest")[0, 0].numpy().astype(np.int64)


def resize_depth_and_masks(
    depth: np.ndarray,
    instance_map: np.ndarray | None,
    semantic_map: np.ndarray | None,
    size: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if size is None:
        return depth.astype(np.float32), instance_map, semantic_map
    return (
        resize_depth_nan_safe(depth, size),
        resize_id_map(instance_map, size),
        resize_id_map(semantic_map, size),
    )


def robust_normalize_depth(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
    z = depth.astype(np.float32).copy()
    if valid.sum() < 10:
        z[:] = 0.0
        return z, 0.0, 1.0
    vals = z[valid]
    med = float(np.median(vals))
    p05, p95 = np.percentile(vals, [5, 95])
    scale = float(max(p95 - p05, 1e-3))
    z = (z - med) / scale
    z[~valid] = 0.0
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, -5.0, 5.0)
    return z.astype(np.float32), med, scale


def depth_to_xyz(depth: np.ndarray, camera: dict[str, Any] | None) -> np.ndarray:
    """Back-project depth into camera coordinates [H,W,3].

    If camera intrinsics are missing, normalized pixel coordinates are used as a
    fallback. This is enough for depth-only shape cues, but metric XYZ should use
    fx/fy/cx/cy when those are available.
    """
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth.astype(np.float32)
    camera = camera or {}
    fx = camera.get("fx")
    fy = camera.get("fy")
    cx = camera.get("cx")
    cy = camera.get("cy")
    if fx is None or fy is None or cx is None or cy is None:
        x = ((xs / max(w - 1, 1)) - 0.5) * z
        y = ((ys / max(h - 1, 1)) - 0.5) * z
    else:
        x = (xs - float(cx)) * z / float(fx)
        y = (ys - float(cy)) * z / float(fy)
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz


def estimate_normals_from_xyz(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Estimate surface normals from camera-coordinate XYZ using finite differences."""
    xyz_clean = np.nan_to_num(xyz.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    xyz_pad = np.pad(xyz_clean, ((1, 1), (1, 1), (0, 0)), mode="edge")
    dx = xyz_pad[1:-1, 2:, :] - xyz_pad[1:-1, :-2, :]
    dy = xyz_pad[2:, 1:-1, :] - xyz_pad[:-2, 1:-1, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-6)
    # Suppress normals touching invalid pixels because NaN background creates artificial boundaries.
    valid_pad = np.pad(valid.astype(bool), ((1, 1), (1, 1)), mode="constant", constant_values=False)
    neigh_valid = (
        valid_pad[1:-1, 1:-1]
        & valid_pad[1:-1, 2:]
        & valid_pad[1:-1, :-2]
        & valid_pad[2:, 1:-1]
        & valid_pad[:-2, 1:-1]
    )
    n[~neigh_valid] = 0.0
    return n.astype(np.float32)


def make_depth_input(
    depth: np.ndarray,
    camera: dict[str, Any] | None = None,
    mode: str = "zv",
) -> np.ndarray:
    """Create depth-only input channels of shape [C,H,W].

    Modes:
      z     : normalized depth
      zv    : normalized depth + valid-depth mask
      xyzv  : normalized camera XYZ + valid-depth mask
      xyznv : normalized camera XYZ + normal XYZ + valid-depth mask

    For the provided dataset, NaN background becomes valid_mask=0 and depth channel=0.
    """
    if mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode}. Choose one of {sorted(VALID_INPUT_MODES)}")
    valid = np.isfinite(depth) & (depth > 0)
    z_norm, med, scale = robust_normalize_depth(depth, valid)
    valid_f = valid.astype(np.float32)

    if mode == "z":
        chans = [z_norm]
    elif mode == "zv":
        chans = [z_norm, valid_f]
    else:
        xyz = depth_to_xyz(depth, camera)
        xyz_norm = np.nan_to_num(xyz.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        xyz_norm[..., 2] = z_norm
        xyz_norm[..., 0] = xyz_norm[..., 0] / max(scale, 1e-3)
        xyz_norm[..., 1] = xyz_norm[..., 1] / max(scale, 1e-3)
        xyz_norm[~valid] = 0.0
        chans = [xyz_norm[..., 0], xyz_norm[..., 1], xyz_norm[..., 2]]
        if mode == "xyznv":
            normals = estimate_normals_from_xyz(xyz, valid)
            chans += [normals[..., 0], normals[..., 1], normals[..., 2]]
        chans.append(valid_f)
    return np.stack(chans, axis=0).astype(np.float32)
