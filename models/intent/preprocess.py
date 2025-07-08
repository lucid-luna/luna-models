# model/intent/preprocess.py

import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils.config import load_config

def main():
    """
    Intent Classification 전처리 스크립트
    """
    config = load_config("intent_config")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    raw_dir = config.data.raw_dir
    proc_dir = config.data.processed_dir
    max_length = config.max_length
    
    os.makedirs(proc_dir, exist_ok=True)
    
    for split in ("train", "validation", "test"):
        print(f"☑️ {split} 데이터셋 전처리 중...")
        ds = load_from_disk(os.path.join(raw_dir, split))
        
        def tokenizer_fn(examples):
            enc = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            enc["labels"] = examples["intent"]
            return enc
        
        ds = ds.map(tokenizer_fn)
        
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        out_dir = os.path.join(proc_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"✅ {split} 데이터셋 전처리 완료, 저장 경로: {out_dir}")
        ds.save_to_disk(out_dir)
        
if __name__ == "__main__":
    main()