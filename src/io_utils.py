from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    keys = ["output_dir", "rectified_dir", "restored_dir", "masks_dir", "crops_dir"]
    for k in keys:
        Path(cfg[k]).mkdir(parents=True, exist_ok=True)


def list_images(input_dir: str | Path) -> List[Path]:
    input_dir = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=_natural_key)


def _natural_key(p: Path):
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def group_images(paths: List[Path], group_by_prefix: bool = True) -> Dict[str, List[Path]]:
    """
    同一页多张照片可以命名为 001_a.jpg、001_b.jpg。
    如果没有下划线后缀，就每张图单独作为一页。
    """
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        key = p.stem
        if group_by_prefix:
            m = re.match(r"^(.+?)[_\-][A-Za-z]+$", p.stem)
            if m:
                key = m.group(1)
        groups.setdefault(key, []).append(p)
    return {k: sorted(v, key=_natural_key) for k, v in sorted(groups.items(), key=lambda kv: _natural_key(Path(kv[0])))}
