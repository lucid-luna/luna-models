# models/emotion/threshold_search.py

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from datasets import load_from_disk
from transformers import AutoTokenizer
from models.emotion.emotion_model import EmotionClassifier
from utils.config import load_config
from safetensors.torch import load_file
import torch

@torch.no_grad()
def main():
    config = load_config("emotion_config")
    model = EmotionClassifier()
    model.load_state_dict(load_file(os.path.join(config.train.output_dir, "model.safetensors")))
    
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    ds = load_from_disk(os.path.join(config.data.processed_dir, "test"))
    
    all_labels = []
    all_logits = []
    
    texts = ds["text"]
    labels = ds["labels"]

    for text, label in tqdm(zip(texts, labels), total=len(texts)):
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device)
        }
        outputs = model(**inputs)
        all_logits.append(outputs["logits"].cpu().numpy()[0])
        all_labels.append(np.array(label))
        
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    
    print("\nThreshold 검색 결과: ")
    best_threshold = 0.0
    best_f1 = 0.0
    for th in np.arange(0.1, 0.9, 0.05):
        preds = (logits > th).astype(int)
        score = f1_score(labels, preds, average="macro")
        print(f"Threshold: {th:.2f} → F1: {score:.4f}")
        if score > best_f1:
            best_f1 = score
            best_threshold = th

    print(f"\n✅ 최적 결과: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
if __name__ == "__main__":
    main()