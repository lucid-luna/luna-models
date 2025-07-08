# motion/emotion/evalutate.py

import os
import torch
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import Trainer, AutoTokenizer, TrainingArguments
from sklearn.metrics import f1_score
from utils.config import load_config
from models.emotion.emotion_model import EmotionClassifier

def main():
    config = load_config("emotion_config")
    
    # 1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = EmotionClassifier()
    model.load_state_dict(load_file(os.path.join(config.train.output_dir, "model.safetensors")))
    model.eval()
    
    # 2) 데이터셋 로드
    eval_ds = load_from_disk(os.path.join(config.data.processed_dir, "test"))
    
    # 3) Metrics 함수 정의
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        elif isinstance(logits, dict):
            logits = logits["logits"]

        preds = (logits > config.inference.threshold).astype(int)
        
        return {
            "f1_macro": f1_score(labels, preds, average="macro")
        }
        
    # 4) Trainer 초기화
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=config.train.output_dir,
            per_device_eval_batch_size=config.train.eval_batch_size,
            dataloader_drop_last=False,
        ),
        compute_metrics=compute_metrics
    )
    
    # 5) 평가
    eval_results = trainer.evaluate(eval_dataset=eval_ds)
    
    print("평가 결과:", eval_results)

if __name__ == "__main__":
    main()