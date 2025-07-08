# models/intent/evaluate.py

import os
import torch
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, TrainingArguments
from sklearn.metrics import accuracy_score
from utils.config import load_config
from models.intent.intent_model import IntentClassifier

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    config = load_config("intent_config")

    # 모델 & 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = IntentClassifier()
    model.load_state_dict(load_file(os.path.join(config.train.output_dir, "model.safetensors")))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)

    # 평가용 데이터셋 로드
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, config.data.test_split))

    # Trainer 설정
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=config.train.output_dir,
            per_device_eval_batch_size=config.train.eval_batch_size,
        ),
        compute_metrics=compute_metrics
    )

    # 평가 실행
    results = trainer.evaluate(eval_dataset=eval_ds)
    
    print("✅ 평가 결과:", results)

if __name__ == "__main__":
    main()
