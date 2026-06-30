from __future__ import annotations

# 必须放在 cv2 / paddle / numpy 大量导入之前，避免 Windows + Anaconda 中 OpenMP 重复加载崩溃。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
from pathlib import Path

import cv2

from src.io_utils import load_config, ensure_dirs, list_images, group_images
from src.geometry import detect_slide_quad, warp_quad_to_canvas, fit_full_image_to_canvas, draw_quad_debug
from src.correction import enhance_courseware
from src.restoration import restore_multi_images
from src.ocr_engine import create_ocr_engine, filter_text_elements
from src.element_extractor import extract_elements
from src.ppt_rebuilder import build_reference_ppt, build_editable_ppt


def process_one_group(group_key: str, paths, cfg, ocr_engine, debug: bool = False):
    canvas_w = int(cfg.get("canvas_width", 1920))
    canvas_h = int(cfg.get("canvas_height", 1080))
    rectified_dir = Path(cfg["rectified_dir"])
    restored_dir = Path(cfg["restored_dir"])
    masks_dir = Path(cfg["masks_dir"])
    crops_dir = Path(cfg["crops_dir"])

    warped_images = []
    statuses = []
    quad_found_count = 0
    confidence_values = []

    for idx, img_path in enumerate(paths):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        quad, det_info = detect_slide_quad(img, cfg["detection"])
        confidence_values.append(float(det_info.get("best_score", 0.0)))
        if quad is not None:
            warped = warp_quad_to_canvas(img, quad, canvas_w, canvas_h)
            status = f"quad_warped:{det_info.get('best_method', 'unknown')}"
            quad_found_count += 1
        else:
            if cfg["detection"].get("fallback_to_full_image", True):
                warped = fit_full_image_to_canvas(img, canvas_w, canvas_h)
                status = "fallback_full_image"
            else:
                raise RuntimeError(f"No slide quadrilateral detected: {img_path}")

        if cfg.get("enhance", {}).get("enable", True):
            warped = enhance_courseware(warped, cfg["enhance"])

        out_img = rectified_dir / f"{group_key}_{idx + 1:02d}_rectified.png"
        cv2.imwrite(str(out_img), warped)
        warped_images.append(warped)
        statuses.append(status)

        if debug:
            dbg_dir = Path(cfg["output_dir"]) / "debug_quads"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            dbg = draw_quad_debug(img, quad, f"{img_path.name} | {status} | score={det_info.get('best_score', 0):.3f}")
            cv2.imwrite(str(dbg_dir / f"{group_key}_{idx + 1:02d}_quad.jpg"), dbg)

    restored, mask, restore_status = restore_multi_images(warped_images, cfg["restoration"])
    restored_path = restored_dir / f"{group_key}_restored.png"
    mask_path = masks_dir / f"{group_key}_mask.png"
    cv2.imwrite(str(restored_path), restored)
    cv2.imwrite(str(mask_path), mask)

    texts = []
    if cfg.get("ocr", {}).get("enable", True):
        raw_texts = ocr_engine.recognize(restored_path, restored)
        texts = filter_text_elements(raw_texts, float(cfg["ocr"].get("min_confidence", 0.45)))

    elems = extract_elements(restored, texts, crops_dir, group_key, cfg["rebuild"])

    row = {
        "slide_key": group_key,
        "input_files": ";".join(str(p.name) for p in paths),
        "num_input_images": len(paths),
        "quad_found_count": quad_found_count,
        "rectify_status": ";".join(statuses),
        "restore_status": restore_status,
        "ocr_text_count": len(texts),
        "graphic_count": len(elems.images),
        "line_count": len(elems.lines),
        "avg_quad_score": round(sum(confidence_values) / max(1, len(confidence_values)), 4),
        "restored_image": str(restored_path),
    }
    return restored_path, elems, row


def main():
    parser = argparse.ArgumentParser(description="课堂拍照课件 → 畸变矫正 → 遮挡恢复 → OCR → 稳定可编辑 PPT")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="输出四边形检测调试图")
    parser.add_argument("--no-ocr", action="store_true", help="临时关闭 OCR，用于先测试几何矫正")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_ocr:
        cfg.setdefault("ocr", {})["enable"] = False
    ensure_dirs(cfg)

    input_paths = list_images(cfg["input_dir"])
    if not input_paths:
        raise RuntimeError(f"No images found in {cfg['input_dir']}")

    groups = group_images(input_paths, bool(cfg.get("group_same_slide_by_prefix", True)))
    print(f"[INFO] Found {len(input_paths)} image(s), grouped into {len(groups)} slide(s).")

    ocr_engine = create_ocr_engine(cfg.get("ocr", {}))

    restored_paths = []
    rebuilt_slides = []
    rows = []

    for i, (group_key, paths) in enumerate(groups.items(), start=1):
        print(f"[INFO] Processing slide {i}/{len(groups)}: {group_key} ({len(paths)} image(s))")
        restored_path, elems, row = process_one_group(group_key, paths, cfg, ocr_engine, debug=args.debug)
        restored_paths.append(restored_path)
        rebuilt_slides.append(elems)
        rows.append(row)

    build_reference_ppt(restored_paths, cfg["reference_pptx"], cfg)
    build_editable_ppt(rebuilt_slides, cfg["editable_pptx"], cfg)

    report_path = Path(cfg["report_csv"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[DONE] Finished.")
    print(f"  Reference image PPT: {cfg['reference_pptx']}")
    print(f"  Editable rebuilt PPT: {cfg['editable_pptx']}")
    print(f"  Report: {cfg['report_csv']}")


if __name__ == "__main__":
    main()
