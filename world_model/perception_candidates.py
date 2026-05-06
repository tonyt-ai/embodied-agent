"""Shared detector/segmenter candidate utilities.

Runtime and offline eval both need the same semantic candidate semantics:
detector boxes, optional segmentation masks, and unmatched `unknown_seg`
regions for static placement targets.
"""

from __future__ import annotations

import numpy as np


def default_bbox_polygon(bbox_norm):
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    return [
        [round(x1, 4), round(y1, 4)],
        [round(x2, 4), round(y1, 4)],
        [round(x2, 4), round(y2, 4)],
        [round(x1, 4), round(y2, 4)],
    ]


def normalize_polygon(points_xy, w: int, h: int, max_points: int = 32):
    if points_xy is None:
        return None
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    if pts.shape[0] > max_points:
        step = max(1, int(round(pts.shape[0] / max_points)))
        pts = pts[::step][:max_points]
    poly = []
    for x, y in pts:
        poly.append([
            round(float(np.clip(x / max(w, 1), 0.0, 1.0)), 4),
            round(float(np.clip(y / max(h, 1), 0.0, 1.0)), 4),
        ])
    return poly


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def dedupe_candidates_nms(candidates, iou_threshold: float = 0.55):
    deduped = []
    for candidate in candidates:
        bbox = candidate.get("bbox")
        label = str(candidate.get("label", "")).lower()
        if not bbox or len(bbox) != 4:
            continue
        keep = True
        for existing in deduped:
            if str(existing.get("label", "")).lower() != label:
                continue
            if bbox_iou(bbox, existing.get("bbox", [0, 0, 0, 0])) >= iou_threshold:
                keep = False
                break
        if keep:
            deduped.append(candidate)
    return deduped


def extract_detector_candidates(results, names, w: int, h: int, conf_min: float = 0.10):
    candidates = []
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if conf < conf_min:
                continue
            label = str(names.get(cls_id, cls_id) if hasattr(names, "get") else names[cls_id])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = (x2 - x1) / max(w, 1)
            bh = (y2 - y1) / max(h, 1)
            if bw < 0.02 or bh < 0.02:
                continue
            bbox_norm = [
                round(float(np.clip(x1 / max(w, 1), 0.0, 1.0)), 4),
                round(float(np.clip(y1 / max(h, 1), 0.0, 1.0)), 4),
                round(float(np.clip(x2 / max(w, 1), 0.0, 1.0)), 4),
                round(float(np.clip(y2 / max(h, 1), 0.0, 1.0)), 4),
            ]
            candidates.append({
                "label": label,
                "bbox": bbox_norm,
                "confidence": round(conf, 4),
                "embedding": None,
                "mask_polygon": default_bbox_polygon(bbox_norm),
                "segmentation_source": "bbox",
            })
    return candidates


def extract_segmentation_candidates(seg_results, w: int, h: int, conf_min: float = 0.08):
    candidates = []
    for result in seg_results or []:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        polys = getattr(masks, "xy", None) if masks is not None else None
        if boxes is None:
            continue
        for idx, box in enumerate(boxes):
            conf = float(box.conf[0].item())
            if conf < conf_min:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = (x2 - x1) / max(w, 1)
            bh = (y2 - y1) / max(h, 1)
            if bw < 0.015 or bh < 0.015:
                continue
            bbox_norm = [
                round(float(np.clip(x1 / max(w, 1), 0.0, 1.0)), 4),
                round(float(np.clip(y1 / max(h, 1), 0.0, 1.0)), 4),
                round(float(np.clip(x2 / max(w, 1), 0.0, 1.0)), 4),
                round(float(np.clip(y2 / max(h, 1), 0.0, 1.0)), 4),
            ]
            poly = None
            if polys is not None and idx < len(polys):
                poly = normalize_polygon(polys[idx], w=w, h=h, max_points=32)
            candidates.append({
                "bbox": bbox_norm,
                "confidence": round(conf, 4),
                "mask_polygon": poly or default_bbox_polygon(bbox_norm),
            })
    return candidates


def associate_segmentation(detector_candidates, segmentation_candidates, min_iou: float = 0.18, source: str = "seg"):
    if not detector_candidates or not segmentation_candidates:
        return detector_candidates
    used = set()
    for det in detector_candidates:
        bbox = det.get("bbox")
        if not bbox:
            continue
        best_idx = -1
        best_iou = 0.0
        for idx, seg in enumerate(segmentation_candidates):
            if idx in used:
                continue
            iou = bbox_iou(bbox, seg.get("bbox", [0, 0, 0, 0]))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= min_iou:
            seg = segmentation_candidates[best_idx]
            det["mask_polygon"] = seg.get("mask_polygon", det.get("mask_polygon"))
            det["segmentation_source"] = source
            det["segmentation_iou"] = round(float(best_iou), 3)
            used.add(best_idx)
    return detector_candidates


def collect_unmatched_segmentation_candidates(
    detector_candidates,
    segmentation_candidates,
    *,
    min_iou: float = 0.18,
    source: str = "seg",
    min_conf: float = 0.15,
    min_area: float = 0.004,
    max_area: float = 0.25,
    max_items: int = 8,
):
    unmatched = []
    for seg in segmentation_candidates or []:
        seg_bbox = seg.get("bbox")
        if not seg_bbox:
            continue
        seg_conf = float(seg.get("confidence", 0.0) or 0.0)
        if seg_conf < min_conf:
            continue
        sx1, sy1, sx2, sy2 = [float(v) for v in seg_bbox]
        area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        if area < min_area or area > max_area:
            continue
        best_iou = 0.0
        for det in detector_candidates or []:
            det_bbox = det.get("bbox")
            if det_bbox:
                best_iou = max(best_iou, bbox_iou(seg_bbox, det_bbox))
        if best_iou >= min_iou:
            continue
        unmatched.append({
            "label": "unknown_seg",
            "bbox": seg_bbox,
            "confidence": round(seg_conf, 4),
            "embedding": None,
            "mask_polygon": seg.get("mask_polygon", default_bbox_polygon(seg_bbox)),
            "segmentation_source": source,
        })
    unmatched = sorted(unmatched, key=lambda it: float(it.get("confidence", 0.0)), reverse=True)
    unmatched = dedupe_candidates_nms(unmatched, iou_threshold=0.45)
    return unmatched[:max(0, int(max_items))]


def build_semantic_candidates(
    frame,
    detector_model,
    *,
    segmentation_model=None,
    device=None,
    detector_conf_min: float = 0.10,
    segmentation_conf_min: float = 0.08,
    segmentation_source: str = "seg",
    add_unmatched: bool = True,
    unmatched_min_conf: float = 0.15,
    unmatched_min_area: float = 0.004,
    unmatched_max_area: float = 0.25,
    unmatched_max_items: int = 8,
):
    kwargs = {"verbose": False}
    if device:
        kwargs["device"] = device
    results = detector_model(frame, **kwargs)
    seg_results = None
    if segmentation_model is not None:
        try:
            seg_results = segmentation_model(frame, **kwargs)
        except Exception:
            seg_results = None
    h, w = frame.shape[:2]
    detector_candidates = extract_detector_candidates(
        results,
        detector_model.names,
        w=w,
        h=h,
        conf_min=detector_conf_min,
    )
    detector_candidates = dedupe_candidates_nms(detector_candidates, iou_threshold=0.55)
    segmentation_candidates = extract_segmentation_candidates(
        seg_results,
        w=w,
        h=h,
        conf_min=segmentation_conf_min,
    )
    semantic_candidates = associate_segmentation(
        detector_candidates,
        segmentation_candidates,
        min_iou=0.18,
        source=segmentation_source,
    )
    if add_unmatched:
        semantic_candidates.extend(collect_unmatched_segmentation_candidates(
            detector_candidates=detector_candidates,
            segmentation_candidates=segmentation_candidates,
            min_iou=0.18,
            source=segmentation_source,
            min_conf=unmatched_min_conf,
            min_area=unmatched_min_area,
            max_area=unmatched_max_area,
            max_items=unmatched_max_items,
        ))
    return dedupe_candidates_nms(
        sorted(semantic_candidates, key=lambda item: item.get("confidence", 0.0), reverse=True),
        iou_threshold=0.55,
    )
