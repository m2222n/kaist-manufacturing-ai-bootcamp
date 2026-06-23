from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_HASH_SUFFIX_RE = re.compile(r"__[0-9a-fA-F]{4,}(?:__[0-9a-fA-F]{4,})*$")


def load_scene_meta(scene_npz: str | Path) -> dict[str, Any]:
    data = np.load(scene_npz, allow_pickle=True)
    if "meta" not in data.files:
        return {}
    raw = data["meta"].item() if getattr(data["meta"], "shape", None) == () else str(data["meta"])
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def canonical_cad_display_name(value: Any) -> str:
    """Return a human-readable CAD name shared by prediction and GT labels.

    The 3D point-cloud pipeline may append generated hash suffixes, e.g.
    ``03_sol_block_front__e79f118d``.  Scene metadata usually stores the STL
    filename, e.g. ``03_sol_block_front.stl``.  For visualization we strip both
    the path/extension and generated suffixes so prediction/GT panels use the
    same visible name.
    """
    if value is None:
        return "cad?"
    name = Path(str(value)).stem
    name = _HASH_SUFFIX_RE.sub("", name)
    # A few older generated ids can contain multiple suffix groups; keep the
    # semantic part before the first double-underscore for display purposes.
    if "__" in name:
        name = name.split("__", 1)[0]
    return name or "cad?"


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Convert metric depth with NaN background into uint8 grayscale visualization.

    Nearer pixels are brighter, matching the provided dataset convention.
    """
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = 1.0 - np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def deterministic_color(index: int) -> tuple[int, int, int]:
    # Bright, stable palette without importing matplotlib.  idx1 starts from
    # palette[0], so use deterministic_color(display_idx - 1).
    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
        (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
        (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    ]
    return palette[index % len(palette)]


def _font(size: int = 12):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def resize_mask_nearest(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    mh, mw = mask.shape
    if (mh, mw) == (h, w):
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), resample=Image.Resampling.NEAREST)
    return np.asarray(img) > 0


def _base_rgb(depth: np.ndarray) -> np.ndarray:
    g = depth_to_vis(depth)
    return np.repeat(g[..., None], 3, axis=2)


def _draw_title(img: Image.Image, title: str) -> Image.Image:
    w, h = img.size
    title_h = 34
    canvas = Image.new("RGB", (w, h + title_h), (20, 20, 20))
    canvas.paste(img, (0, title_h))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), title, fill=(255, 255, 255), font=_font(14))
    return canvas


def _visible_instance_order(inst: np.ndarray, meta: dict[str, Any]) -> list[int]:
    present = {int(v) for v in np.unique(inst) if int(v) != 0}
    visible = meta.get("visible_inst_ids", []) if isinstance(meta, dict) else []
    ordered: list[int] = []
    for v in visible:
        try:
            inst_id = int(v)
        except Exception:
            continue
        if inst_id in present and inst_id not in ordered:
            ordered.append(inst_id)
    for inst_id in sorted(present):
        if inst_id not in ordered:
            ordered.append(inst_id)
    return ordered


def build_gt_display_objects(depth: np.ndarray, scene_npz: str | Path) -> list[dict[str, Any]]:
    data = np.load(scene_npz, allow_pickle=True)
    inst = data["inst_id"].astype(np.int64)
    cat = data["category_id"].astype(np.int64)
    meta = load_scene_meta(scene_npz)
    instances = meta.get("instances", {}) if isinstance(meta, dict) else {}
    objects: list[dict[str, Any]] = []
    for display_idx, inst_id in enumerate(_visible_instance_order(inst, meta), start=1):
        mask = inst == inst_id
        if not mask.any():
            continue
        m = instances.get(str(inst_id), instances.get(inst_id, {})) if isinstance(instances, dict) else {}
        stl = m.get("stl", "") if isinstance(m, dict) else ""
        cid = int(np.bincount(cat[mask].ravel()).argmax()) if mask.any() else -1
        ys, xs = np.where(mask)
        objects.append({
            "idx": display_idx,
            "inst_id": inst_id,
            "class_id": cid,
            "cad_name": canonical_cad_display_name(stl),
            "mask": mask,
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "color": deterministic_color(display_idx - 1),
        })
    return objects


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(float(union), 1.0))


def build_prediction_display_objects(
    preds: list[dict[str, Any]],
    masks: np.ndarray,
    target_hw: tuple[int, int],
    gt_objects: list[dict[str, Any]] | None = None,
    match_iou_thresh: float = 0.05,
) -> list[dict[str, Any]]:
    """Create unified display metadata for predicted objects.

    If GT objects are available, each prediction receives the idx of the GT
    object with the highest mask IoU.  This makes the middle and right panels
    use the same ``idxN`` naming and color for corresponding objects.  If no GT
    is available or no IoU match is found, predictions fall back to rank-based
    idx values.
    """
    out: list[dict[str, Any]] = []
    gt_objects = gt_objects or []
    next_unmatched_idx = len(gt_objects) + 1
    for n, pred in enumerate(preds):
        pred_mask = None
        if masks.ndim == 3 and n < len(masks):
            pred_mask = resize_mask_nearest(masks[n].astype(bool), target_hw)

        best_gt = None
        best_iou = 0.0
        if pred_mask is not None and gt_objects:
            for gt in gt_objects:
                iou = _mask_iou(pred_mask, gt["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        if best_gt is not None and best_iou >= match_iou_thresh:
            idx = int(best_gt["idx"])
            color = best_gt["color"]
            matched_iou = best_iou
        else:
            # In deployment visualization there is no GT panel; rank-based idx
            # is still clearer than query id.  If GT is shown but this prediction
            # does not overlap any GT sufficiently, give it a new idx after the
            # GT range so it is visibly unmatched.
            idx = n + 1 if not gt_objects else next_unmatched_idx
            if gt_objects:
                next_unmatched_idx += 1
            color = deterministic_color(idx - 1)
            matched_iou = None

        out.append({
            "idx": idx,
            "color": color,
            "cad_name": canonical_cad_display_name(pred.get("cad_id", pred.get("cad_index", "cad?"))),
            "class_id": pred.get("class_id", "?"),
            "score": float(pred.get("score", 0.0)),
            "cad_score": float(pred.get("cad_score", 0.0)) if "cad_score" in pred else 0.0,
            "matched_iou": matched_iou,
        })
    return out


def draw_prediction_panel(
    base_rgb: np.ndarray,
    preds: list[dict[str, Any]],
    masks: np.ndarray,
    title: str,
    display_objects: list[dict[str, Any]] | None = None,
) -> Image.Image:
    h, w = base_rgb.shape[:2]
    overlay = base_rgb.astype(np.float32).copy()
    img = Image.fromarray(base_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(11)
    display_objects = display_objects or build_prediction_display_objects(preds, masks, (h, w))

    # First blend masks into an RGB array.
    for n, pred in enumerate(preds):
        disp = display_objects[n] if n < len(display_objects) else {"color": deterministic_color(n)}
        color = np.array(disp["color"], dtype=np.float32)
        if n < len(masks):
            mask = resize_mask_nearest(masks[n].astype(bool), (h, w))
            overlay[mask] = overlay[mask] * 0.62 + color * 0.38
    img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for n, pred in enumerate(preds):
        disp = display_objects[n] if n < len(display_objects) else build_prediction_display_objects([pred], masks[n:n + 1], (h, w))[0]
        color = disp["color"]
        x1, y1, x2, y2 = [float(v) for v in pred["bbox_xyxy"]]
        pred_h = pred.get("image_height") or pred.get("pred_image_height")
        pred_w = pred.get("image_width") or pred.get("pred_image_width")
        if pred_h and pred_w:
            sx = w / float(pred_w)
            sy = h / float(pred_h)
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
        elif len(masks) > n and masks[n].ndim == 2:
            mh, mw = masks[n].shape
            if (mh, mw) != (h, w):
                x1, x2 = x1 * (w / mw), x2 * (w / mw)
                y1, y2 = y1 * (h / mh), y2 * (h / mh)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        draw.rectangle([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)], outline=color, width=3)
        label = f"idx{int(disp['idx']):02d} C{disp['class_id']}\n{disp['cad_name']}\nS={disp['score']:.2f} CAD={disp['cad_score']:.2f}"
        tx, ty = x1, max(y1 - 43, 0)
        # Text background for readability.
        bbox = draw.multiline_textbbox((tx, ty), label, font=font, spacing=1)
        draw.rectangle(bbox, fill=tuple(int(c * 0.65) for c in color))
        draw.multiline_text((tx, ty), label, fill=(255, 255, 255), font=font, spacing=1)
    return _draw_title(img, title)


def draw_gt_panel(depth: np.ndarray, scene_npz: str | Path, title: str, gt_objects: list[dict[str, Any]] | None = None) -> Image.Image:
    base = _base_rgb(depth)
    overlay = base.astype(np.float32).copy()
    gt_objects = gt_objects or build_gt_display_objects(depth, scene_npz)
    for gt in gt_objects:
        color = np.array(gt["color"], dtype=np.float32)
        overlay[gt["mask"]] = overlay[gt["mask"]] * 0.62 + color * 0.38
    img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(11)
    for gt in gt_objects:
        x1, y1, x2, y2 = [int(v) for v in gt["bbox_xyxy"]]
        color = gt["color"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"idx{int(gt['idx']):02d} C{gt['class_id']}\n{gt['cad_name']}"
        tx, ty = x1, max(y1 - 30, 0)
        bbox = draw.multiline_textbbox((tx, ty), label, font=font, spacing=1)
        draw.rectangle(bbox, fill=tuple(int(c * 0.65) for c in color))
        draw.multiline_text((tx, ty), label, fill=(255, 255, 255), font=font, spacing=1)
    return _draw_title(img, title)


def save_prediction_visualization(
    *,
    depth: np.ndarray,
    predictions: list[dict[str, Any]],
    masks: np.ndarray,
    out_path: str | Path,
    scene_npz: str | Path | None = None,
    include_gt: bool = False,
    title: str = "Inference: detected instances + CAD decision",
) -> None:
    base = _base_rgb(depth)
    h, w = depth.shape
    if masks.ndim == 3 and masks.shape[0] > 0:
        mh, mw = masks.shape[1:]
        for p in predictions:
            p.setdefault("pred_image_height", int(mh))
            p.setdefault("pred_image_width", int(mw))

    gt_objects = None
    if include_gt and scene_npz is not None:
        gt_objects = build_gt_display_objects(depth, scene_npz)
    pred_display = build_prediction_display_objects(predictions, masks, (h, w), gt_objects=gt_objects)

    panels = [
        _draw_title(Image.fromarray(base).convert("RGB"), "Input depth, nearer = brighter"),
        draw_prediction_panel(base, predictions, masks, title, display_objects=pred_display),
    ]
    if include_gt and scene_npz is not None:
        panels.append(draw_gt_panel(depth, scene_npz, "Ground truth overlay", gt_objects=gt_objects))
    panel_w = max(p.width for p in panels)
    panel_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h), (20, 20, 20))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * panel_w, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
