from __future__ import annotations

import numpy as np


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=float)


def box_center(box: np.ndarray) -> tuple[float, float]:
    """Centre de la boîte (le centre bas est préférable pour les véhicules, mais
    le centre géométrique reste plus stable quand la hauteur de boîte varie)."""
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def box_area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU pairwise entre deux ensembles de boîtes (N,4) et (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float)[:, None, :]
    b = np.asarray(b, dtype=float)[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(dets: np.ndarray, iou_thresh: float) -> np.ndarray:
    """NMS par classe sur un tableau (N,6) = x1,y1,x2,y2,conf,cls."""
    if dets.size == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 4])]
    keep: list[int] = []
    for cls in np.unique(dets[:, 5]):
        idx = np.where(dets[:, 5] == cls)[0]
        while len(idx):
            i = idx[0]
            keep.append(int(i))
            if len(idx) == 1:
                break
            rest = idx[1:]
            ious = iou_matrix(dets[i : i + 1, :4], dets[rest, :4]).ravel()
            idx = rest[ious < iou_thresh]
    return dets[np.array(sorted(keep), dtype=int)]


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.asarray(boxes, dtype=float).copy()
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h - 1)
    return out
