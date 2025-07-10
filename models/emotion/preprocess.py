# ====================================================================
#  File: models/emotion/preprocess.py
# ====================================================================
"""
LunaEmotion 데이터 전처리 스크립트

이 스크립트는 학습용 원시 감정 데이터셋을 불러와 토크나이즈하고,
모델 학습에 적합한 형식으로 변환하여 저장합니다.

실행 예시:
    python -m models.emotion.preprocess
"""

import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils.config import load_config

def main():
    config = load_config("emotion_config")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    raw_base = config.data.raw_dir
    proc_base = config.data.processed_dir
    max_length = config.max_length
    
    label_list = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]

    for split in ("train", "validation", "test"):
        print(f"[L.U.N.A] {split} 데이터셋 전처리 중...")
        ds = load_from_disk(os.path.join(raw_base, split))
        
        def tokenizer_fn(examples):
            enc = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            
            def one_hot_label(label_ids):
                vec = [0] * len(label_list)
                for i in label_ids:
                    if 0 <= i < len(label_list):
                        vec[i] = 1
                return vec
            
            enc["labels"] = [one_hot_label(label_ids) for label_ids in examples["labels"]]
            return enc
        
        ds = ds.map(
            tokenizer_fn,
            batched=True,
            remove_columns=[col for col in ds.column_names if col not in ("text",)]
        )
        
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        print("예시 라벨:", ds[0]["labels"])
        
        out_dir = os.path.join(proc_base, split)
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"[L.U.N.A.] {split} 데이터셋이 {out_dir}에 저장되었습니다.")
        
if __name__ == "__main__":
    main()