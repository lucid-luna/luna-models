# model/intent/train.py

import os
import numpy as np
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score
from utils.config import load_config
from models.intent.intent_model import IntentClassifier

if __name__ == "__main__":
    config = load_config("intent_config")
    
    # 1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    model = IntentClassifier()
    
    # 2) 데이터셋 로드
    train_ds = load_from_disk(os.path.join(config.data.processed_dir, "train"))
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, "validation"))
    
    # 3) Metrics 함수 정의
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, dict)):
            logits = logits[0]
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(np.array(labels), np.array(preds))
        }
    
    # 4) TrainingArguments
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
    
    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # 6) 모델 학습
    trainer.train()
    trainer.save_model(config.train.output_dir)
    tokenizer.save_pretrained(config.train.output_dir)
    print(f"모델과 토크나이저가 {config.train.output_dir}에 저장되었습니다.")