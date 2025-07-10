# ====================================================================
#  File: models/multiintent/evaluate.py
# ====================================================================
"""
LunaMultiIntent 모델 평가 스크립트

이 스크립트는 학습된 MultiIntentClassifier 모델을 로드하고 테스트 
데이터셋에 대해 평가를 수행한 뒤 F1 매크로, 마이크로, 샘플별 점수를 출력합니다.

실행 예시:
    python -m models.multiintent.evaluate
"""

import os
import torch
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, TrainingArguments
from sklearn.metrics import f1_score, classification_report
from utils.config import load_config
from models.multiintent.intent_model import MultiIntentClassifier

def compute_metrics(eval_pred):
    """
    Trainer.evaluate에서 사용되는 메트릭 함수

    Args:
        eval_pred: (logits, labels)
            logits: 모델 출력 로짓, shape=(batch_size, num_labels)
            labels: 정답 원-핫 벡터, shape=(batch_size, num_labels)

    Returns:
        dict: f1_macro, f1_micro, f1_samples
    """
    logits, labels = eval_pred
    
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)  # thresholding
    
    labels = labels.astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_samples": f1_score(labels, preds, average="samples", zero_division=0),
    }

def main():
    config = load_config("multiintent_config")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = MultiIntentClassifier(config.model.name, num_labels=config.model.num_labels)
    model_path = os.path.join(config.train.output_dir, "model.safetensors")
    model.load_state_dict(load_file(model_path, device=device.type))
    model.to(device)
    model.eval()
    print(f"[L.U.N.A] MultiIntent 모델 로딩 완료. 추론 장치: {device.type.upper()}")

    test_split = config.data.test_split
    eval_ds = load_from_disk(os.path.join(
        config.data.processed_dir,
        test_split
    ))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=config.train.output_dir,
            per_device_eval_batch_size=config.train.eval_batch_size,
        ),
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=eval_ds)
    
    print("[L.U.N.A.] 평가 결과:", results)

if __name__ == "__main__":
    main()
