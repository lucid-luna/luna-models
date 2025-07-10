# ====================================================================
#  File: models/vision/preprocess.py
# ====================================================================
"""
LunaVision 전처리 스크립트

이 스크립트는 이미지 및 텍스트 데이터를 전처리하여 모델 학습에 적합한 형식으로 변환합니다.

실행 예시:
    python -m models.vision.preprocess --config <path_to_config>
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset, Audio, Features, Image, Value, Dataset, DatasetDict
from PIL import Image as PILImage
from tqdm import tqdm

from utils.config import load_config_dict as load_config

# ----------
# Helper functions
# ----------

def sha256_file(path: Path) -> str:
    """SHA-256 해시를 계산합니다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _resize_and_save(img_path: str, out_dir: Path, size: int = 512) -> str:
    """이미지를 지정된 크기로 리사이즈하고 저장합니다."""
    img = PILImage.open(img_path).convert("RGB")
    img.thumbnail((size, size), PILImage.Resampling.LANCZOS)
    sha = sha256_file(Path(img_path))[:16]
    out_file = out_dir / f"{sha}.jpg"
    img.save(out_file, quality=90)
    return str(out_file)

# ----------
# Core Preprocessing function
# ----------

def build_dataset(cfg: Dict, out_base: Path) -> None:
    """
    Iterate over dataset specs and save split-wise Arrow files.

    cfg example:
        { "name": "coco_caption",
          "hf_id": "coco_captions",
          "splits": ["train", "validation"],
          "img_key": "image",
          "txt_key": "caption",
          "prompt_tpl": "<img>\\nUSER: Describe the image.\\nASSISTANT:",
          "answer_key": "caption"}
    """
    spec_list: List[Dict] = cfg["datasets"]
    for spec in spec_list:
        hf_id   = spec["hf_id"]
        splits  = spec["splits"]
        img_k   = spec["img_key"]
        txt_k   = spec["txt_key"]
        prompt  = spec["prompt_tpl"]
        answer_k = spec.get("answer_key", txt_k)
        out_dir = out_base / spec["name"]
        
        for split in splits:
            print(f"{hf_id}[{split}] 을 처리 중입니다...")
            ds = load_dataset(hf_id, split=split)
            out_img_dir = out_dir / "images" / split
            out_img_dir.mkdir(parents=True, exist_ok=True)
            
            def _map(ex):
                new_path = _resize_and_save(ex[img_k]["path"], out_img_dir)
                
                return {
                    "image": new_path,
                    "prompt": prompt,
                    "answer": ex[answer_k]
                }
            
            ds_proc = ds.map(_map, remove_columns=ds.column_names,
                             desc=f"Resize {split}", num_proc=4)
            
            features = Features({
                "image": Image(),
                "prompt": Value("string"),
                "answer": Value("string")
            })
            ds_proc = ds_proc.cast(features)
            tgt = out_dir / f"{split}.arrow"
            ds_proc.save_to_disk(str(tgt))
            print(f" {len(ds_proc)}개의 항목을 {tgt}에 저장했습니다.")
            
# --------------------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LunaVision Dataset Preprocessor")
    ap.add_argument("--config", required=True, help="vision_config.yaml")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    build_dataset(cfg, output_root)
    
if __name__ == "__main__":
    main()