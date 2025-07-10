# ====================================================================
#  File: models/intent/train.py
# ====================================================================
"""
LunaIntent 모델 학습 스크립트

이 스크립트는 전처리된 intent 데이터셋을 로드하고 IntentClassifier 
모델을 초기화한 뒤 학습 및 검증을 수행하고 최적 모델과 토크나이저를 저장합니다.

실행 예시:
    python -m models.intent.train
"""

import os
import numpy as np

from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score

from utils.config import load_config
from models.intent.intent_model import IntentClassifier

def main():
    config = load_config("intent_config")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = IntentClassifier()
    
    train_ds = load_from_disk(os.path.join(config.data.processed_dir, "train"))
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, "validation"))
    
    def compute_metrics(eval_pred):
        """
        eval_pred: (logits, labels)
        logits은 (batch_size, num_labels) 형태, labels은 (batch_size,) 형태
        """
        logits, labels = eval_pred
        if isinstance(logits, (tuple, dict)):
            logits = logits[0]
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
    
    training_args = TrainingArguments(
        output_dir=config.train.output_dir,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.train_batch_size,
        per_device_eval_batch_size=config.train.eval_batch_size,
        learning_rate=config.train.learning_rate,
        eval_strategy=config.train.eval_strategy,
        save_strategy=config.train.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=config.train.best_metric,
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=os.path.join(config.train.output_dir, "logs"),
        logging_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(config.train.output_dir)
    tokenizer.save_pretrained(config.train.output_dir)
    print(f"[L.U.N.A] 학습 완료. 모델과 토크나이저가 '{config.train.output_dir}'에 저장되었습니다.")
    
if __name__ == "__main__":
    main()