#!/usr/bin/env python
"""
실루엣(binary mask) 추출.

렌더 이미지는 흰 배경(bg=255) + 밝은 회색 부품이라 단순 임계값으로
배경/전경 분리가 가능하다 (SAM 같은 무거운 분할 불필요 = 우리 데이터의 강점).
부품이 배경보다 어둡고 음영이 있으므로 "배경에 가까운 흰색"을 배경으로 본다.
"""
import numpy as np
from PIL import Image


def load_silhouette(path, bg_thresh=245):
    """
    이미지 경로 -> binary mask (H,W) bool. True=부품(전경).

    bg_thresh: 세 채널 모두 이 값 이상이면 배경(흰색)으로 간주.
    렌더 배경은 255, 부품은 baseColor 0.82~0.88 * 조명이라 보통 200~240 →
    245 임계값이면 부품을 배경으로 오인하지 않는다. 부품 가장자리 하이라이트가
    날아가는 경우만 주의 (필요 시 임계값 낮춤).
    """
    img = np.asarray(Image.open(path).convert("RGB"))
    is_bg = np.all(img >= bg_thresh, axis=2)
    mask = ~is_bg
    return mask


def mask_bbox(mask):
    """전경 bounding box (rmin, rmax, cmin, cmax). 비었으면 None."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)
