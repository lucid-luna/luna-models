# model/intent/train.py

import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from utils.config import load_config
from models.multiintent.intent_model import MultiIntentClassifier
import torch
import torch.nn as nn

def compute_metrics(eval_pred):
    """
    평가 결과를 기반으로 F1, Precision, Recall을 계산하는 함수
    """
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    labels = np.array(labels)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    config = load_config("multiintent_config")
    
    # 1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    # 2) 데이터셋 로드
    train_ds = load_from_disk(os.path.join(config.data.processed_dir, "train"))
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, "validation"))
    
    # 3) 라벨 계산
    with open(os.path.join(config.data.processed_dir, "label_list.json"), "r", encoding="utf-8") as f:
        label_list = json.load(f)
    num_labels = len(label_list)
    
    model = MultiIntentClassifier(config.model.name, num_labels)
    
    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=config.train.output_dir,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.train_batch_size,
        per_device_eval_batch_size=config.train.eval_batch_size,
        learning_rate=config.train.learning_rate,
        evaluation_strategy=config.train.eval_strategy,
        save_strategy=config.train.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=config.train.best_metric,
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=os.path.join(config.train.output_dir, "logs"),
        logging_steps=50,
    )
    
    # 5) 손실 함수 정의
    def luna_loss(model, inputs, return_outputs=False):
        """
        L.U.N.A. 손실 함수 정의
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, labels.float())
        return (loss, outputs) if return_outputs else loss

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_fn=luna_loss,
    )
    
    # 6) 모델 학습
    trainer.train()
    trainer.save_model(config.train.output_dir)
    tokenizer.save_pretrained(config.train.output_dir)
    
    print(f"모델과 토크나이저가 {config.train.output_dir}에 저장되었습니다.")