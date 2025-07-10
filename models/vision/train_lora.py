# ====================================================================
#  File: models/vision/train_lora.py
# ====================================================================
"""
LunaVision LoRA 파인튜닝 스크립트

실행 예시:
    python -m models.vision.train_lora \
        --config config/vision_config.yaml \
        --output_dir outputs/vision/fp16_lora
"""

from __future__ import annotations

import argparse, json, os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_from_disk, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments
)

from utils.config import load_config_dict as load_config

# ----------
# Data Helper
# ----------

def load_arrow_dataset(path: Path, splits: List[str]) -> DatasetDict:
    """'train.arrow', 'validation.arrow' 로 저장된 Arrow → DatasetDict"""
    return DatasetDict({sp: load_from_disk(path / f"{sp}.arrow") for sp in splits})

def add_vision_tokens(ds, processor, max_len: int):
    """이미지 / 텍스트 -> pixel_values / input_ids / labels 매핑"""
    def _map(ex):
        enc = processor(
            images=ex["image"],
            text=f'{ex["prompt"]} {ex["answer"]}',
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        ex.update({k: v[0] for k, v in enc.items()})
        ex["labels"] = ex["input_ids"].clone()
        return ex

    cols_to_remove = ds.column_names
    return ds.map(
        _map,
        remove_columns=cols_to_remove,
        desc="tokenize+vision",
        num_proc=os.cpu_count(),
    )
    
def vlm_collate(batch):
    """pixel_values / input_ids / labels → 배치 단위로 묶기"""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels
    }

# ----------
# LoRA 학습 파이프라인
# ----------

def train(cfg_path: str, output_dir: str):
    cfg = load_config(cfg_path)["train_lora"]
    
    # 1) 모델 및 토크나이저 로드
    base_model = cfg["model_name"]
    processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    
    # 2) LoRA 설정
    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    
    # 3) 데이터셋 로드 및 토큰화
    raw_ds = load_arrow_dataset(Path(cfg["dataset_dir"]), ["train", "validation"])
    max_len = cfg.get("max_length", 512)
    proc_ds = DatasetDict(
        {
            split: add_vision_tokens(dset, processor, max_len)
            for split, dset in raw_ds.items()
        }
    )
    
    # 4) Trainer 설정
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("train_bsz", 2),
        per_device_eval_batch_size=cfg.get("eval_bsz", 2),
        gradient_accumulation_steps=cfg.get("grad_acc", 8),
        num_train_epochs=cfg.get("epochs", 3),
        learning_rate=cfg.get("lr", 1e-4),
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.get("logging_steps", 50),
        save_total_limit=2,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=proc_ds["train"],
        eval_dataset=proc_ds.get("validation"),
        data_collator=vlm_collate,
        tokenizer=processor,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # 6) LoRA 설정 저장
    with open(Path(output_dir) / "lora_config.json", "w") as f:
        json.dump(
            {
                "base_model": base_model,
                "lora_config": lora_cfg.to_dict(),
                "output_dir": output_dir,
            },
            f,
            indent=2,
        )
    
# ----------
# CLI Entry Point
# ----------

def main():
    ap = argparse.ArgumentParser(description="LunaVision LoRA Trainer")
    ap.add_argument("--config", required=True, help="vision_config.yaml")
    ap.add_argument("--output_dir", required=True, help="체크포인트 저장 경로")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args.config, args.output_dir)


if __name__ == "__main__":
    main()