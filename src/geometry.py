from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def resize_keep_aspect(img: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, float(long_side) / float(max(h, w)))
    if scale == 1.0:
        return img.copy(), 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def polygon_area(quad: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32).reshape(-1, 2))))


def _quad_from_rect(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _rectangularity(quad: np.ndarray) -> float:
    area = polygon_area(quad)
    rect = cv2.minAreaRect(quad.astype(np.float32))
    rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
    return min(area / rect_area, 1.0)


def _aspect_score(quad: np.ndarray, target: float = 16 / 9) -> float:
    q = order_points(quad)
    w1 = np.linalg.norm(q[1] - q[0])
    w2 = np.linalg.norm(q[2] - q[3])
    h1 = np.linalg.norm(q[3] - q[0])
    h2 = np.linalg.norm(q[2] - q[1])
    ww, hh = max(1.0, (w1 + w2) / 2), max(1.0, (h1 + h2) / 2)
    ar = ww / hh
    # 允许 4:3 到 16:9，中间也不重罚。越接近 target 越高。
    return float(np.exp(-abs(np.log(ar / target)) / 0.55))


def _inside_outside_contrast(img: np.ndarray, quad: np.ndarray) -> Tuple[float, float]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, order_points(quad).astype(np.int32), 255)
    inside = gray[mask > 0]
    outside = gray[mask == 0]
    if len(inside) == 0:
        return 0.0, 0.0
    mean_inside = float(np.mean(inside))
    if len(outside) == 0:
        return mean_inside, 0.0
    mean_outside = float(np.mean(outside))
    return mean_inside, abs(mean_inside - mean_outside)


def quad_score(quad: np.ndarray, img: np.ndarray, method: str, cfg: Dict[str, Any]) -> float:
    h, w = img.shape[:2]
    area_ratio = polygon_area(quad) / float(w * h + 1e-6)
    if area_ratio < float(cfg.get("min_area_ratio", 0.04)):
        return -1.0
    if area_ratio > 0.98:
        return -1.0

    rect_score = _rectangularity(quad)
    aspect = _aspect_score(quad, float(cfg.get("target_aspect", 16 / 9)))
    mean_inside, contrast = _inside_outside_contrast(img, quad)
    bright_score = min(mean_inside / 210.0, 1.0)
    contrast_score = min(contrast / 70.0, 1.0)
    area_score = min(area_ratio / float(cfg.get("ideal_area_ratio", 0.55)), 1.0)

    # 贴边不再强惩罚。很多真实照片里课件区域会贴近照片边界。
    border_penalty = 0.0
    margin = int(0.005 * max(w, h))
    q = order_points(quad)
    if np.any(q[:, 0] <= margin) or np.any(q[:, 0] >= w - margin) or np.any(q[:, 1] <= margin) or np.any(q[:, 1] >= h - margin):
        border_penalty = 0.03

    method_bonus = {"bright_quad": 0.08, "bright_minrect": 0.04, "edge_quad": 0.03, "edge_minrect": 0.0}.get(method, 0.0)
    return 0.31 * area_score + 0.22 * rect_score + 0.18 * aspect + 0.14 * bright_score + 0.12 * contrast_score + method_bonus - border_penalty


def _candidate_from_contours(contours, method_prefix: str, min_area: float, eps_ratios: List[float]) -> List[Tuple[str, np.ndarray, float]]:
    candidates: List[Tuple[str, np.ndarray, float]] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        found = False
        for eps in eps_ratios:
            approx = cv2.approxPolyDP(cnt, float(eps) * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append((f"{method_prefix}_quad", order_points(approx.reshape(4, 2)), area))
                found = True
                break
        if not found:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            candidates.append((f"{method_prefix}_minrect", order_points(box), area))
            x, y, ww, hh = cv2.boundingRect(cnt)
            candidates.append((f"{method_prefix}_bbox", _quad_from_rect(x, y, ww, hh), area))
    return candidates


def _edge_candidates(small: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[str, np.ndarray, float]]:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = int(cfg.get("gaussian_kernel", 5))
    if k % 2 == 0:
        k += 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, int(cfg.get("canny_low", 50)), int(cfg.get("canny_high", 150)))
    mk = int(cfg.get("morph_kernel", 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    edges = cv2.dilate(edges, kernel, iterations=int(cfg.get("dilate_iter", 2)))
    edges = cv2.erode(edges, kernel, iterations=int(cfg.get("erode_iter", 1)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = small.shape[:2]
    min_area = float(cfg.get("min_area_ratio", 0.04)) * w * h
    return _candidate_from_contours(contours, "edge", min_area, list(cfg.get("eps_ratios", [0.015, 0.02, 0.03, 0.05])))


def _bright_candidates(small: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[str, np.ndarray, float]]:
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (5, 5), 0)
    # 对投影屏/白底 PPT，亮区通常就是课件主体；用 Otsu + 分位阈值共同尝试。
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    qthr = int(np.percentile(blur, float(cfg.get("bright_percentile", 72))))
    _, qmask = cv2.threshold(blur, qthr, 255, cv2.THRESH_BINARY)
    masks = [otsu, qmask, cv2.bitwise_or(otsu, qmask)]

    all_candidates: List[Tuple[str, np.ndarray, float]] = []
    min_area = float(cfg.get("min_area_ratio", 0.04)) * w * h
    for mask in masks:
        mk = int(cfg.get("bright_morph_kernel", 21))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_candidates.extend(_candidate_from_contours(contours, "bright", min_area, list(cfg.get("eps_ratios", [0.015, 0.02, 0.03, 0.05]))))
    return all_candidates


def detect_slide_quad(img: np.ndarray, det_cfg: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    small, scale = resize_keep_aspect(img, int(det_cfg.get("resize_long_side", 1400)))
    candidates = []
    candidates.extend(_bright_candidates(small, det_cfg))
    candidates.extend(_edge_candidates(small, det_cfg))

    # 去重：按整数化四点坐标去掉近似重复。
    dedup = []
    seen = set()
    for method, quad, area in candidates:
        q = order_points(quad)
        key = tuple(np.round(q.reshape(-1) / 8).astype(int).tolist())
        if key not in seen:
            seen.add(key)
            dedup.append((method, q, area))

    scored = []
    for method, quad, area in dedup:
        score = quad_score(quad, small, method, det_cfg)
        if score >= float(det_cfg.get("min_accept_score", 0.38)):
            scored.append((score, method, quad, area))

    debug = {"scale": scale, "num_candidates": len(dedup), "num_accepted": len(scored)}
    if not scored:
        return None, debug
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_method, best_quad_small, best_area = scored[0]
    best_quad = order_points(best_quad_small / scale)
    debug.update({"best_score": float(best_score), "best_method": best_method, "best_area": float(best_area)})
    return best_quad, debug


def warp_quad_to_canvas(img: np.ndarray, quad: np.ndarray, width: int, height: int) -> np.ndarray:
    src = order_points(quad).astype(np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def fit_full_image_to_canvas(img: np.ndarray, width: int, height: int, bg=(255, 255, 255)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_quad_debug(img: np.ndarray, quad: Optional[np.ndarray], text: str = "") -> np.ndarray:
    out = img.copy()
    if quad is not None:
        q = order_points(quad).astype(int)
        cv2.polylines(out, [q], True, (0, 0, 255), 5)
        for i, p in enumerate(q):
            cv2.circle(out, tuple(p), 10, (0, 255, 255), -1)
            cv2.putText(out, str(i + 1), tuple(p + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    if text:
        cv2.putText(out, text[:80], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return out
