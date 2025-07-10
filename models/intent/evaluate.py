# ====================================================================
#  File: models/intent/evaluate.py
# ====================================================================
"""
LunaIntent 모델 평가 스크립트

학습된 IntentClassifier 모델을 로드하고 지정된
테스트 데이터셋에 대해 평가를 수행한 후 정확도(accuracy)를 출력합니다.

실행 예시:
    python -m models.intent.evaluate
"""

import os
import torch
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, TrainingArguments
from sklearn.metrics import accuracy_score
from utils.config import load_config
from models.intent.intent_model import IntentClassifier

def compute_metrics(eval_pred):
    """
    Trainer.evaluate에서 사용하는 메트릭 함수

    Args:
        eval_pred: (logits, labels)
            logits: 모델 출력 로짓, shape=(batch_size, num_labels)
            labels: 정답 레이블, shape=(batch_size,)

    Returns:
        dict: accuracy 키로 정확도 반환
    """
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    config = load_config("intent_config")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = IntentClassifier()
    model.load_state_dict(load_file(os.path.join(config.train.output_dir, "model.safetensors")))
    model.to(device)    
    model.eval()

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
