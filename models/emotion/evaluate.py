# ====================================================================
#  File: models/emotion/evaluate.py
# ====================================================================
"""
LunaEmotion 모델 평가 스크립트

학습이 완료된 EmotionClassifier 모델의 성능을 평가합니다.
테스트 데이터셋에 대한 F1-macro 점수를 계산하여 출력합니다.

실행 예시:
    python -m models.emotion.evaluate
"""

import os
import torch
import numpy as np
from scipy.special import expit  # 시그모이드 함수
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, TrainingArguments
from sklearn.metrics import f1_score
from utils.config import load_config
from models.emotion.emotion_model import EmotionClassifier

def compute_metrics(eval_pred: tuple) -> dict:
    """
    예측 결과를 기반으로 F1-macro 점수를 계산하는 함수
    
    Args:
        eval_pred (tuple): 모델의 예측 로짓과 실제 레이블을 담은 튜플

    Returns:
        dict: "f1_macro" 키에 F1 macro 점수를 담은 딕셔너리
    """
    config = load_config("emotion_config")
    logits, labels = eval_pred
    
    probs = expit(logits)
    
    preds = (probs > config.inference.threshold).astype(int)
    
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)
    }

def main():
    config = load_config("emotion_config")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = EmotionClassifier()
    
    model_path = os.path.join(config.train.output_dir, "model.safetensors")
    model.load_state_dict(load_file(model_path))
    model.eval()
    
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, "test"))
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=config.train.output_dir,
            per_device_eval_batch_size=config.train.eval_batch_size,
            dataloader_drop_last=False,
            remove_unused_columns=False,
        ),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    eval_results = trainer.evaluate(eval_dataset=eval_ds)
    
    print(f"  - Loss: {eval_results['eval_loss']:.4f}")
    print(f"  - F1 (Macro): {eval_results['eval_f1_macro']:.4f}")

if __name__ == "__main__":
    main()