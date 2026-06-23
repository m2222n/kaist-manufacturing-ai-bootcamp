#!/usr/bin/env python3
"""
Convert a folder of STL meshes into a point-cloud dataset for CAD encoders.

Output per CAD:
  - <cad_id>.npz
      points         : (N, 3) canonical points, centered and uniformly scaled
      normals        : (N, 3) surface normals
      features       : (N, 6) [points, normals]
      points_raw     : (N, 3) sampled points in the original CAD / assembly coordinate
      nocs           : (N, 3) bbox-normalized coordinates in [0, 1] per axis
      sample_source  : (N,) 0=surface sample, 1=sharp-edge-biased sample
      center         : (3,) bbox center used for canonical transform
      scale          : scalar max bbox extent used for canonical transform
      bbox_min       : (3,)
      bbox_max       : (3,)
      extents        : (3,)
  - <cad_id>.json metadata
  - optional <cad_id>.ply preview point cloud

Also writes manifest.json in the output directory.

Example:
  python stl_to_pointcloud_dataset.py \
      --input_dir ./stl_folder \
      --output_dir ./pc_dataset \
      --n_points 8192 \
      --edge_ratio 0.2 \
      --export_ply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm


def safe_id_from_path(path: Path, root: Path) -> str:
    """Create a filesystem-safe, stable CAD id from a relative STL path."""
    rel = path.relative_to(root).with_suffix("").as_posix()
    rel = re.sub(r"[^0-9A-Za-z가-힣._/-]+", "_", rel)
    rel = rel.replace("/", "__")
    # Add a short hash to avoid collisions when same names exist in subfolders.
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{rel}__{digest}"


def load_as_mesh(path: Path) -> trimesh.Trimesh:
    """
    Load STL or scene as a single Trimesh.
    STL often stores duplicated vertices per triangle; process_mesh() welds them.
    """
    loaded = trimesh.load(path, force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported geometry type from {path}: {type(loaded)}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    return mesh


def _call_if_exists(obj, name: str, *args, **kwargs):
    fn = getattr(obj, name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def process_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Basic cleanup: remove invalid faces, weld duplicated vertices, recompute normals."""
    mesh = mesh.copy()

    # Remove NaN / Inf vertices if present.
    vertices = np.asarray(mesh.vertices)
    valid_vertices = np.isfinite(vertices).all(axis=1)
    if not valid_vertices.all():
        # Keep faces whose three vertices are valid.
        valid_faces = valid_vertices[np.asarray(mesh.faces)].all(axis=1)
        mesh.update_faces(valid_faces)
        mesh.remove_unreferenced_vertices()

    # Remove degenerate / duplicate faces when the installed trimesh supports it.
    try:
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        else:
            _call_if_exists(mesh, "remove_degenerate_faces")
    except Exception:
        pass

    try:
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        else:
            _call_if_exists(mesh, "remove_duplicate_faces")
    except Exception:
        pass

    # Vertex welding: merge duplicated vertices with identical / near-identical coordinates.
    # This reconstructs adjacency/topology that raw STL may not explicitly encode.
    try:
        mesh.merge_vertices()
    except TypeError:
        mesh.merge_vertices(radius=None)

    mesh.remove_unreferenced_vertices()

    # Fix / recompute normals.
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except Exception:
        pass
    _ = mesh.face_normals
    _ = mesh.vertex_normals

    return mesh


def sample_surface_points(
    mesh: trimesh.Trimesh,
    count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted surface sampling. Returns points, normals, source labels."""
    if count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # sample_surface is area-weighted and returns the face index for each sampled point.
    points, face_idx = trimesh.sample.sample_surface(mesh, count=count)
    normals = np.asarray(mesh.face_normals)[face_idx]
    source = np.zeros((len(points),), dtype=np.int64)
    return points.astype(np.float32), normals.astype(np.float32), source


def sample_sharp_edge_points(
    mesh: trimesh.Trimesh,
    count: int,
    sharp_angle_deg: float = 35.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points along sharp edges to avoid losing small holes, slots, ribs, and chamfers.
    Returns points, approximate normals, source labels.
    """
    if count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    vertices = np.asarray(mesh.vertices)
    face_normals = np.asarray(mesh.face_normals)

    edge_vertices: Optional[np.ndarray] = None
    edge_normals: Optional[np.ndarray] = None

    # Primary: use face adjacency angle to detect sharp edges.
    try:
        angles = np.asarray(mesh.face_adjacency_angles)
        adj = np.asarray(mesh.face_adjacency)
        adj_edges = np.asarray(mesh.face_adjacency_edges)
        threshold = math.radians(sharp_angle_deg)
        keep = angles >= threshold
        if keep.any():
            edge_vertices = adj_edges[keep]
            n0 = face_normals[adj[keep, 0]]
            n1 = face_normals[adj[keep, 1]]
            edge_normals = n0 + n1
            norm = np.linalg.norm(edge_normals, axis=1, keepdims=True)
            edge_normals = edge_normals / np.maximum(norm, 1e-8)
    except Exception:
        edge_vertices = None
        edge_normals = None

    # Fallback: boundary edges if available. Many watertight CAD meshes have none.
    if edge_vertices is None or len(edge_vertices) == 0:
        try:
            boundary = np.asarray(mesh.edges_boundary)
            if len(boundary) > 0:
                edge_vertices = boundary
                edge_normals = np.zeros((len(boundary), 3), dtype=np.float32)
                edge_normals[:, 2] = 1.0
        except Exception:
            pass

    # Final fallback: no useful sharp edges found, use normal surface sampling.
    if edge_vertices is None or len(edge_vertices) == 0:
        return sample_surface_points(mesh, count)

    v0 = vertices[edge_vertices[:, 0]]
    v1 = vertices[edge_vertices[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    valid = lengths > 1e-12
    if not valid.any():
        return sample_surface_points(mesh, count)

    edge_vertices = edge_vertices[valid]
    edge_normals = edge_normals[valid]
    v0 = vertices[edge_vertices[:, 0]]
    v1 = vertices[edge_vertices[:, 1]]
    lengths = lengths[valid]
    probs = lengths / lengths.sum()

    edge_idx = np.random.choice(len(edge_vertices), size=count, replace=True, p=probs)
    t = np.random.random((count, 1)).astype(np.float32)
    points = v0[edge_idx] * (1.0 - t) + v1[edge_idx] * t
    normals = edge_normals[edge_idx]

    # If fallback boundary normals were dummy, overwrite them with nearest face normal when cheap enough.
    # Keeping approximate normals is usually acceptable for edge-biased auxiliary samples.
    source = np.ones((count,), dtype=np.int64)
    return points.astype(np.float32), normals.astype(np.float32), source


def sample_mixed_points(
    mesh: trimesh.Trimesh,
    n_points: int,
    edge_ratio: float,
    sharp_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix area-weighted surface samples and sharp-edge-biased samples."""
    edge_ratio = float(np.clip(edge_ratio, 0.0, 0.8))
    n_edge = int(round(n_points * edge_ratio))
    n_surface = n_points - n_edge

    p_s, n_s, src_s = sample_surface_points(mesh, n_surface)
    p_e, n_e, src_e = sample_sharp_edge_points(mesh, n_edge, sharp_angle_deg)

    points = np.concatenate([p_s, p_e], axis=0)
    normals = np.concatenate([n_s, n_e], axis=0)
    source = np.concatenate([src_s, src_e], axis=0)

    # If edge fallback returned surface source labels, force correct total but not source semantics.
    if len(points) != n_points:
        missing = n_points - len(points)
        p_m, n_m, src_m = sample_surface_points(mesh, missing)
        points = np.concatenate([points, p_m], axis=0)
        normals = np.concatenate([normals, n_m], axis=0)
        source = np.concatenate([source, src_m], axis=0)

    # Shuffle so edge samples are not clustered at the end.
    perm = np.random.permutation(len(points))[:n_points]
    return points[perm], normals[perm], source[perm]


def normalize_points(
    points_raw: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Uniform canonical normalization and bbox NOCS normalization.
      points = (points_raw - bbox_center) / max_extent
      nocs   = (points_raw - bbox_min) / bbox_extent_per_axis
    """
    bbox_min = bbox_min.astype(np.float32)
    bbox_max = bbox_max.astype(np.float32)
    extents = bbox_max - bbox_min
    center = (bbox_min + bbox_max) * 0.5
    scale = float(np.max(extents))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Invalid mesh scale; bbox has near-zero size")

    points = (points_raw - center[None, :]) / scale
    nocs = (points_raw - bbox_min[None, :]) / np.maximum(extents[None, :], 1e-12)
    nocs = np.clip(nocs, 0.0, 1.0)
    return points.astype(np.float32), center.astype(np.float32), scale, nocs.astype(np.float32)


def export_pointcloud_ply(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    """Export canonical point cloud preview as PLY."""
    cloud = trimesh.PointCloud(vertices=points)
    # trimesh PointCloud does not always retain normals in exported PLY; points are enough for preview.
    cloud.export(path)


def process_one_file(
    stl_path: Path,
    input_root: Path,
    output_dir: Path,
    n_points: int,
    edge_ratio: float,
    sharp_angle_deg: float,
    export_ply: bool,
    class_from: str,
) -> Dict:
    raw_mesh = load_as_mesh(stl_path)
    raw_vertices_count = int(len(raw_mesh.vertices))
    raw_faces_count = int(len(raw_mesh.faces))

    mesh = process_mesh(raw_mesh)

    bbox_min, bbox_max = np.asarray(mesh.bounds, dtype=np.float32)
    extents = bbox_max - bbox_min

    points_raw, normals, sample_source = sample_mixed_points(
        mesh=mesh,
        n_points=n_points,
        edge_ratio=edge_ratio,
        sharp_angle_deg=sharp_angle_deg,
    )
    points, center, scale, nocs = normalize_points(points_raw, bbox_min, bbox_max)

    # Normalize normals defensively.
    normals = normals.astype(np.float32)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    features = np.concatenate([points, normals], axis=1).astype(np.float32)

    cad_id = safe_id_from_path(stl_path, input_root)
    class_name = stl_path.parent.name if class_from == "parent" else stl_path.stem

    npz_path = output_dir / f"{cad_id}.npz"
    json_path = output_dir / f"{cad_id}.json"
    ply_path = output_dir / f"{cad_id}.ply"

    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
        features=features.astype(np.float32),
        points_raw=points_raw.astype(np.float32),
        nocs=nocs.astype(np.float32),
        sample_source=sample_source.astype(np.int64),
        center=center.astype(np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        bbox_min=bbox_min.astype(np.float32),
        bbox_max=bbox_max.astype(np.float32),
        extents=extents.astype(np.float32),
    )

    metadata = {
        "cad_id": cad_id,
        "class_name": class_name,
        "input_path": str(stl_path),
        "npz_path": str(npz_path),
        "raw_vertices_count_before_welding": raw_vertices_count,
        "raw_faces_count": raw_faces_count,
        "processed_vertices_count_after_welding": int(len(mesh.vertices)),
        "processed_faces_count": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number) if np.isfinite(mesh.euler_number) else None,
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "bbox_center": center.astype(float).tolist(),
        "bbox_extents": extents.astype(float).tolist(),
        "scale_max_extent": float(scale),
        "n_points": int(n_points),
        "edge_ratio_requested": float(edge_ratio),
        "edge_sample_count_actual": int((sample_source == 1).sum()),
        "sharp_angle_deg": float(sharp_angle_deg),
        "canonical_transform": {
            "formula": "points = (points_raw - center) / scale",
            "center": center.astype(float).tolist(),
            "scale": float(scale),
        },
        "raw_from_canonical_transform": {
            "formula": "points_raw = points * scale + center",
            "center": center.astype(float).tolist(),
            "scale": float(scale),
        },
        "nocs_transform": {
            "formula": "nocs = (points_raw - bbox_min) / (bbox_max - bbox_min)",
            "bbox_min": bbox_min.astype(float).tolist(),
            "bbox_max": bbox_max.astype(float).tolist(),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if export_ply:
        export_pointcloud_ply(ply_path, points, normals)
        metadata["ply_path"] = str(ply_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def find_stl_files(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.stl" if recursive else "*.stl"
    files = sorted(input_dir.glob(pattern))
    # Also handle uppercase extensions.
    if recursive:
        files += sorted(input_dir.glob("**/*.STL"))
    else:
        files += sorted(input_dir.glob("*.STL"))
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert STL folder to CAD point-cloud dataset")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing STL files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dataset folder")
    parser.add_argument("--n_points", type=int, default=8192, help="Number of points per CAD")
    parser.add_argument("--edge_ratio", type=float, default=0.2, help="Ratio of samples placed on sharp edges")
    parser.add_argument("--sharp_angle_deg", type=float, default=35.0, help="Face angle threshold for sharp edges")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--recursive", action="store_true", help="Search STL files recursively")
    parser.add_argument("--export_ply", action="store_true", help="Export canonical point cloud PLY previews")
    parser.add_argument(
        "--class_from",
        choices=["stem", "parent"],
        default="stem",
        help="Use file stem or parent folder as class_name in manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_files = find_stl_files(input_dir, recursive=args.recursive)
    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {input_dir}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_points": int(args.n_points),
        "edge_ratio": float(args.edge_ratio),
        "sharp_angle_deg": float(args.sharp_angle_deg),
        "seed": int(args.seed),
        "items": [],
    }

    errors = []
    for stl_path in tqdm(stl_files, desc="Converting STL to point clouds"):
        try:
            item = process_one_file(
                stl_path=stl_path,
                input_root=input_dir,
                output_dir=output_dir,
                n_points=args.n_points,
                edge_ratio=args.edge_ratio,
                sharp_angle_deg=args.sharp_angle_deg,
                export_ply=args.export_ply,
                class_from=args.class_from,
            )
            manifest["items"].append(item)
        except Exception as e:
            errors.append({"path": str(stl_path), "error": repr(e)})

    manifest["errors"] = errors
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Converted {len(manifest['items'])}/{len(stl_files)} STL files.")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err['path']}: {err['error']}")


if __name__ == "__main__":
    main()
