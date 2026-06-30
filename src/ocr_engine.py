from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class TextElement:
    x: int
    y: int
    w: int
    h: int
    text: str
    conf: float
    color: Tuple[int, int, int] = (0, 0, 0)  # RGB


def _estimate_text_color(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    x, y, w, h = box
    H, W = img_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return (0, 0, 0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 背景亮时取暗像素，背景暗时取亮像素。
    p15 = np.percentile(gray, 15)
    p85 = np.percentile(gray, 85)
    if np.mean(gray) < 120:
        pix = roi[gray >= p85 - 5]
    else:
        pix = roi[gray <= p15 + 8]
    if len(pix) == 0:
        return (0, 0, 0)
    bgr = np.median(pix, axis=0)
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


class BaseOCREngine:
    def recognize(self, image_path: str | Path, img_bgr: np.ndarray) -> List[TextElement]:
        raise NotImplementedError


class NoOCREngine(BaseOCREngine):
    def recognize(self, image_path: str | Path, img_bgr: np.ndarray) -> List[TextElement]:
        return []


class PaddleOCREngine(BaseOCREngine):
    def __init__(self, lang: str = "ch", version: str = "auto"):
        from paddleocr import PaddleOCR
        self.api_version = "2.x"
        # 先尝试 2.x 稳定接口。你的 Windows + Python3.8 环境建议 paddlepaddle==2.6.2 + paddleocr==2.7.3。
        try:
            kwargs = {"use_angle_cls": True, "lang": lang, "show_log": False}
            if version and version.lower() not in ("auto", "default"):
                kwargs["ocr_version"] = version
            self.ocr = PaddleOCR(**kwargs)
            self.api_version = "2.x"
            print("[INFO] PaddleOCR 初始化成功：2.x 接口。")
            return
        except TypeError:
            pass

        # 兼容 PaddleOCR 3.x。若你环境里是 3.x，会走这里。
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self.api_version = "3.x"
        print("[INFO] PaddleOCR 初始化成功：3.x 接口。")

    def recognize(self, image_path: str | Path, img_bgr: np.ndarray) -> List[TextElement]:
        if self.api_version == "3.x":
            return self._recognize_v3(image_path, img_bgr)
        return self._recognize_v2(image_path, img_bgr)

    def _recognize_v2(self, image_path: str | Path, img_bgr: np.ndarray) -> List[TextElement]:
        result = self.ocr.ocr(str(image_path), cls=True)
        elems: List[TextElement] = []
        if not result:
            return elems
        lines = result[0] if isinstance(result, list) and len(result) > 0 else []
        for line in lines:
            if len(line) < 2:
                continue
            box = np.array(line[0], dtype=np.float32)
            txt_conf = line[1]
            text = str(txt_conf[0])
            conf = float(txt_conf[1])
            self._append_text(elems, img_bgr, box, text, conf)
        return elems

    def _recognize_v3(self, image_path: str | Path, img_bgr: np.ndarray) -> List[TextElement]:
        result = self.ocr.predict(str(image_path))
        elems: List[TextElement] = []
        for page in result:
            data = getattr(page, "json", None)
            if callable(data):
                data = page.json
            if not isinstance(data, dict):
                try:
                    data = page.to_dict()
                except Exception:
                    data = {}
            res = data.get("res", data)
            texts = res.get("rec_texts", []) or []
            scores = res.get("rec_scores", []) or []
            polys = res.get("rec_polys", res.get("dt_polys", [])) or []
            for text, conf, poly in zip(texts, scores, polys):
                box = np.array(poly, dtype=np.float32).reshape(-1, 2)
                self._append_text(elems, img_bgr, box, str(text), float(conf))
        return elems

    @staticmethod
    def _append_text(elems: List[TextElement], img_bgr: np.ndarray, box: np.ndarray, text: str, conf: float) -> None:
        text = text.strip()
        if not text:
            return
        x0, y0 = np.floor(box.min(axis=0)).astype(int)
        x1, y1 = np.ceil(box.max(axis=0)).astype(int)
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        # 过滤过小碎片，减少“页码/噪点/错识别”的影响。
        if w < 6 or h < 6:
            return
        color = _estimate_text_color(img_bgr, (x0, y0, w, h))
        elems.append(TextElement(int(x0), int(y0), int(w), int(h), text, float(conf), color))


class TesseractOCREngine(BaseOCREngine):
    def __init__(self, lang: str = "chi_sim+eng"):
        import pytesseract
        self.pytesseract = pytesseract
        self.lang = "chi_sim+eng" if lang in ("ch", "cn", "zh") else lang
        print("[INFO] Tesseract OCR 初始化成功。")

    def recognize(self, image_path: str | Path, img_bgr: np.ndarray) -> List[TextElement]:
        from pytesseract import Output
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = self.pytesseract.image_to_data(rgb, lang=self.lang, output_type=Output.DICT)
        elems: List[TextElement] = []
        for i, text in enumerate(data.get("text", [])):
            text = (text or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except Exception:
                conf = 0.0
            x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
            color = _estimate_text_color(img_bgr, (x, y, w, h))
            elems.append(TextElement(x, y, w, h, text, conf, color))
        return elems


def create_ocr_engine(cfg: Dict[str, Any]) -> BaseOCREngine:
    if not cfg.get("enable", True):
        print("[INFO] OCR 已关闭。")
        return NoOCREngine()
    engine = str(cfg.get("engine", "auto")).lower()
    lang = str(cfg.get("language", "ch"))
    version = str(cfg.get("paddle_ocr_version", "auto"))

    if engine in ("none", "off", "false"):
        return NoOCREngine()

    if engine in ("auto", "paddle"):
        try:
            return PaddleOCREngine(lang=lang, version=version)
        except Exception as e:
            msg = f"[WARN] PaddleOCR 初始化失败：{type(e).__name__}: {e}"
            if engine == "paddle":
                raise RuntimeError(msg) from e
            print(msg)

    if engine in ("auto", "tesseract"):
        try:
            return TesseractOCREngine(lang=lang)
        except Exception as e:
            msg = f"[WARN] Tesseract OCR 初始化失败：{type(e).__name__}: {e}"
            if engine == "tesseract":
                raise RuntimeError(msg) from e
            print(msg)

    print("[WARN] 没有可用 OCR 引擎，将生成视觉稳定版 PPT，但文字不会变成可编辑文本框。")
    return NoOCREngine()


def filter_text_elements(elems: List[TextElement], min_conf: float) -> List[TextElement]:
    out: List[TextElement] = []
    for e in elems:
        if e.conf < min_conf:
            continue
        if e.w <= 2 or e.h <= 2 or not e.text.strip():
            continue
        out.append(e)
    return out
