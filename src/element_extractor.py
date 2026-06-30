from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .ocr_engine import TextElement


@dataclass
class ImageElement:
    x: int
    y: int
    w: int
    h: int
    path: str


@dataclass
class LineElement:
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (0, 0, 0)
    width: float = 1.0


@dataclass
class SlideElements:
    texts: List[TextElement]
    images: List[ImageElement]
    lines: List[LineElement]
    bg_color: Tuple[int, int, int]
    background_image: str | None = None


def estimate_background_color(img: np.ndarray) -> Tuple[int, int, int]:
    """估计页面边缘背景色，用于极端情况下的纯色背景。返回 RGB。"""
    h, w = img.shape[:2]
    pad = max(8, min(h, w) // 60)
    border = np.concatenate([
        img[:pad, :, :].reshape(-1, 3),
        img[-pad:, :, :].reshape(-1, 3),
        img[:, :pad, :].reshape(-1, 3),
        img[:, -pad:, :].reshape(-1, 3),
    ], axis=0)
    bgr = np.median(border, axis=0)
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int, pad: int = 0) -> Tuple[int, int, int, int]:
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(W, int(x + w) + pad)
    y1 = min(H, int(y + h) + pad)
    return x0, y0, x1, y1


def mask_text_boxes(shape: Tuple[int, int], texts: List[TextElement], padding: int = 5) -> np.ndarray:
    """粗文本框 mask。box_fill 模式会用这个。"""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for t in texts:
        x0, y0, x1, y1 = _clip_box(t.x, t.y, t.w, t.h, W, H, padding)
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def _median_bg_from_ring(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, ring: int = 8) -> np.ndarray:
    """从文本框周边取一圈像素估计局部背景色。返回 BGR。"""
    H, W = img.shape[:2]
    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(W, x1 + ring), min(H, y1 + ring)
    outer = img[ry0:ry1, rx0:rx1]
    if outer.size == 0:
        return np.array([255, 255, 255], dtype=np.float32)
    mask = np.ones(outer.shape[:2], dtype=np.uint8)
    ix0, iy0 = x0 - rx0, y0 - ry0
    ix1, iy1 = x1 - rx0, y1 - ry0
    if 0 <= ix0 < ix1 <= mask.shape[1] and 0 <= iy0 < iy1 <= mask.shape[0]:
        mask[iy0:iy1, ix0:ix1] = 0
    pix = outer[mask > 0]
    if len(pix) < 20:
        pix = outer.reshape(-1, 3)
    return np.median(pix, axis=0).astype(np.float32)


def _stroke_mask_for_text(img: np.ndarray, text: TextElement, cfg: Dict[str, Any]) -> np.ndarray:
    """
    对单个 OCR 文本框生成“文字笔画 mask”。
    不直接整块抹掉文本框，而是找与局部背景差异较大的像素，尽量只清除字形，保留图片/背景。
    """
    H, W = img.shape[:2]
    pad = int(cfg.get("text_inpaint_padding_px", 3))
    x0, y0, x1, y1 = _clip_box(text.x, text.y, text.w, text.h, W, H, pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((H, W), dtype=np.uint8)

    roi = img[y0:y1, x0:x1]
    bg = _median_bg_from_ring(img, x0, y0, x1, y1, ring=max(6, pad * 3))

    # 颜色距离：能处理黑字、红字、蓝字、灰字。
    diff = np.linalg.norm(roi.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    # 灰度距离：对低饱和文字也有效。
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = float(np.dot(bg, [0.114, 0.587, 0.299]))
    gray_diff = np.abs(gray - bg_gray)

    thr = float(cfg.get("text_diff_threshold", 24))
    local_mask = ((diff > thr) | (gray_diff > max(15, thr * 0.65))).astype(np.uint8) * 255

    # 限制极端情况：如果整块都被选中，说明背景估计失败，退回到只取最暗/最亮或高饱和像素。
    selected_ratio = float(np.count_nonzero(local_mask)) / float(local_mask.size)
    if selected_ratio > 0.55:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        if bg_gray > 145:
            local_mask = ((gray < np.percentile(gray, 38)) | (sat > 80)).astype(np.uint8) * 255
        else:
            local_mask = ((gray > np.percentile(gray, 62)) | (sat > 80)).astype(np.uint8) * 255

    # 清理噪点并轻微膨胀，覆盖抗锯齿边缘。
    k = max(1, int(cfg.get("text_mask_dilate_px", 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    local_mask = cv2.dilate(local_mask, kernel, iterations=1)

    full = np.zeros((H, W), dtype=np.uint8)
    full[y0:y1, x0:x1] = local_mask
    return full


def build_text_clean_mask(img: np.ndarray, texts: List[TextElement], cfg: Dict[str, Any]) -> np.ndarray:
    """生成用于清除底图原文字的 mask。"""
    H, W = img.shape[:2]
    if not texts:
        return np.zeros((H, W), dtype=np.uint8)
    method = str(cfg.get("text_clean_method", "stroke_inpaint")).lower()
    if method == "box_fill":
        return mask_text_boxes((H, W), texts, int(cfg.get("box_fill_padding_px", 1)))

    mask = np.zeros((H, W), dtype=np.uint8)
    for t in texts:
        # 过滤非常大的框，避免误把整张截图当文字清掉。
        if t.w * t.h > 0.08 * W * H:
            continue
        mask = cv2.bitwise_or(mask, _stroke_mask_for_text(img, t, cfg))
    return mask


def make_clean_background(img: np.ndarray, texts: List[TextElement], cfg: Dict[str, Any]) -> np.ndarray:
    """
    从矫正后的完整课件图中清掉已识别的原文字，之后再叠加可编辑文本框。
    这样可以避免“底图原文字 + OCR文字框”重复。
    """
    method = str(cfg.get("text_clean_method", "stroke_inpaint")).lower()
    mask = build_text_clean_mask(img, texts, cfg)
    if np.count_nonzero(mask) == 0:
        return img.copy()

    if method == "box_fill":
        # 去重最强：整块文本框用局部背景色填充。缺点是可能有矩形色块。
        bg = img.copy()
        H, W = img.shape[:2]
        pad = int(cfg.get("box_fill_padding_px", 1))
        for t in texts:
            if t.w * t.h > 0.08 * W * H:
                continue
            x0, y0, x1, y1 = _clip_box(t.x, t.y, t.w, t.h, W, H, pad)
            color = _median_bg_from_ring(img, x0, y0, x1, y1, ring=10)
            bg[y0:y1, x0:x1] = color.astype(np.uint8)
        return bg

    # 默认：只修补文字笔画，尽量保留底图中的图片、渐变、线条。
    radius = int(cfg.get("text_inpaint_radius", 3))
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def make_background_image(img: np.ndarray, texts: List[TextElement], crops_dir: str | Path, slide_key: str, cfg: Dict[str, Any]) -> str:
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get("background_mode", "clean_overlay")).lower()

    if mode == "clean_overlay" and texts:
        bg = make_clean_background(img, texts, cfg)
        suffix = "clean_background"
    else:
        # visual_overlay 会保留原始文字，叠加 OCR 文本时必然重复。
        bg = img.copy()
        suffix = "visual_background"

    out_path = crops_dir / f"{slide_key}_{suffix}.png"
    cv2.imwrite(str(out_path), bg)

    # 同时保存 mask 便于调试。没有 texts 时不生成。
    if mode == "clean_overlay" and texts:
        mask = build_text_clean_mask(img, texts, cfg)
        cv2.imwrite(str(crops_dir / f"{slide_key}_text_clean_mask.png"), mask)
    return str(out_path)


def _line_color(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int]:
    H, W = img.shape[:2]
    x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
    y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    pix = img[mask > 0]
    if len(pix) == 0:
        return (0, 0, 0)
    bgr = np.median(pix, axis=0)
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def extract_lines(img: np.ndarray, texts: List[TextElement], cfg: Dict[str, Any]) -> List[LineElement]:
    """线条检测默认关闭，避免文字边缘/截图边框被误判成大量 PPT 线条。"""
    if not cfg.get("detect_lines", False):
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    text_mask = mask_text_boxes((H, W), texts, int(cfg.get("text_box_padding_px", 4)) + 2)
    edges[text_mask > 0] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(cfg.get("line_threshold", 120)),
        minLineLength=int(cfg.get("line_min_length_px", 260)),
        maxLineGap=int(cfg.get("line_max_gap_px", 8)),
    )
    if lines is None:
        return []

    elems: List[LineElement] = []
    max_lines = int(cfg.get("max_lines", 60))
    for l in lines[:300]:
        x1, y1, x2, y2 = map(int, l[0])
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length < int(cfg.get("line_min_length_px", 260)):
            continue
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if not (dy <= 0.04 * max(dx, 1) or dx <= 0.04 * max(dy, 1)):
            continue
        color = _line_color(img, x1, y1, x2, y2)
        elems.append(LineElement(x1, y1, x2, y2, color, float(cfg.get("line_width_pt", 0.75))))
        if len(elems) >= max_lines:
            break
    return elems


def extract_elements(img: np.ndarray, texts: List[TextElement], crops_dir: str | Path, slide_key: str, cfg: Dict[str, Any]) -> SlideElements:
    bg = estimate_background_color(img)
    background_image = make_background_image(img, texts, crops_dir, slide_key, cfg)
    lines = extract_lines(img, texts, cfg)
    # 复杂图片/截图/图表不再拆碎，统一留在背景图中，防止排版崩坏。
    images: List[ImageElement] = []
    return SlideElements(texts=texts, images=images, lines=lines, bg_color=bg, background_image=background_image)
