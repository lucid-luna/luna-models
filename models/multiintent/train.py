# ====================================================================
#  File: models/multiintent/train.py
# ====================================================================
"""
LunaMultiIntent 모델 학습 스크립트

실행 예시:
    python -m models.multiintent.train
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

from utils.config import load_config
from models.multiintent.intent_model import MultiIntentClassifier

# ----------
# Helper functions
# ----------

def compute_metrics(eval_pred):
    """
    micro-averaged Precision, Recall, F1 점수 계산

    Args:
        eval_pred: (logits, labels) 튜플
    Returns:
        dict: precision, recall, f1
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

# ----------
# Loss Function
# ----------
def luna_loss(model, inputs, return_outputs=False):
    """
    Trainer가 사용하는 손실 함수 구현

    Args:
        model: MultiIntentClassifier
        inputs: dict, 'labels' 포함
        return_outputs: loss와 모델 출력을 함께 반환할지 여부
    Returns:
        loss or (loss, outputs)
    """
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(outputs["logits"], labels.float())
    return (loss, outputs) if return_outputs else loss

def main():
    config = load_config("multiintent_config")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    label_list_path = os.path.join(config.data.processed_dir, "label_list.json")
    with open(label_list_path, "r", encoding="utf-8") as f:
        label_list = json.load(f)
    num_labels = len(label_list)
    model = MultiIntentClassifier(
        model_name=config.model.name,
        num_labels=num_labels
    )
    
    train_ds = load_from_disk(os.path.join(config.data.processed_dir, "train"))
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, "validation"))
    
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
        compute_metrics=compute_metrics,
        loss_fn=luna_loss,
    )
    
    trainer.train()
    trainer.save_model(config.train.output_dir)
    tokenizer.save_pretrained(config.train.output_dir)
    
    print(f"[L.U.N.A] 학습 완료! 모델과 토크나이저가 '{config.train.output_dir}'에 저장되었습니다.")
    
if __name__ == "__main__":
    main()