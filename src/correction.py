from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def gray_world_white_balance(img: np.ndarray, strength: float = 0.65) -> np.ndarray:
    img_f = img.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / (means + 1e-6)
    balanced = img_f * scale.reshape(1, 1, 3)
    out = img_f * (1.0 - strength) + balanced * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_courseware(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    out = img.copy()
    if cfg.get("white_balance", True):
        out = gray_world_white_balance(out, float(cfg.get("white_balance_strength", 0.55)))

    if cfg.get("clahe", True):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        grid = int(cfg.get("clahe_tile_grid", 8))
        clahe = cv2.createCLAHE(clipLimit=float(cfg.get("clahe_clip_limit", 1.6)), tileGridSize=(grid, grid))
        l2 = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    if cfg.get("denoise", True):
        # 轻微降噪，避免 OCR 把墙面噪声和屏幕颗粒当成文字。
        out = cv2.bilateralFilter(out, d=5, sigmaColor=30, sigmaSpace=30)

    if cfg.get("sharpen", True):
        blur = cv2.GaussianBlur(out, (0, 0), float(cfg.get("sharpen_sigma", 1.0)))
        out = cv2.addWeighted(out, float(cfg.get("sharpen_amount", 1.25)), blur, float(cfg.get("sharpen_blur_weight", -0.25)), 0)
    return out
