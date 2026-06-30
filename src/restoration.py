from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def detect_dark_occlusion(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    单张图只能检测明显暗色遮挡，例如人影、手臂。
    不能真实恢复被遮挡文字，只能做背景修补。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = int(cfg.get("dark_occlusion_threshold", 45))
    mask = (gray < threshold).astype(np.uint8) * 255

    # 避免把黑字误判为遮挡：只保留较大连通域。
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    min_area = float(cfg.get("min_occlusion_area_ratio", 0.004)) * h * w
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area >= min_area and ww > 25 and hh > 25:
            clean[labels == i] = 255
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    return clean


def restore_single_image(img: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    if not cfg.get("single_image_inpaint", True):
        return img, np.zeros(img.shape[:2], dtype=np.uint8), "single_no_inpaint"
    mask = detect_dark_occlusion(img, cfg)
    if int(mask.sum()) == 0:
        return img, mask, "single_no_occlusion_detected"
    restored = cv2.inpaint(img, mask, int(cfg.get("inpaint_radius", 5)), cv2.INPAINT_TELEA)
    return restored, mask, "single_inpaint_background_only"


def restore_multi_images(images: List[np.ndarray], cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    if len(images) == 0:
        raise ValueError("No images to restore")
    if len(images) == 1:
        return restore_single_image(images[0], cfg)

    if cfg.get("use_multi_image_median_fusion", True):
        stack = np.stack(images, axis=0).astype(np.uint8)
        median = np.median(stack, axis=0).astype(np.uint8)
        # 与中值图差异大的区域视为遮挡/异常区域。
        diffs = np.max(np.abs(stack.astype(np.int16) - median.astype(np.int16)), axis=3)
        mask = (np.max(diffs, axis=0) > int(cfg.get("diff_threshold", 45))).astype(np.uint8) * 255
        return median, mask, "multi_image_median_fusion"

    return images[0], np.zeros(images[0].shape[:2], dtype=np.uint8), "multi_disabled_use_first"
