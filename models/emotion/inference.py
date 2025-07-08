# model/emotion/inference.py

import torch
import argparse
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils.config import load_config
from models.emotion.emotion_model import EmotionClassifier

def inter(text: str):
    config = load_config("emotion_config")
    
    device = torch.device("cpu")
    
    # 1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    model = EmotionClassifier()
    model.load_state_dict(load_file(f"{config.train.output_dir}/model.safetensors"))
    model.to(device)
    model.eval()

    # 2) 입력 텍스트 토크나이징
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt"
    )
    
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }
    
    # 3) 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs["probs"].squeeze(0)
        preds = (probs > config.inference.threshold).int().tolist()
        
    # 4) 라벨 매핑
    labels = config.inference.label_list
    predicted_labels = [label for label, bit in zip(labels, preds) if bit == 1]
    
    # 5) 결과 출력
    print(f"입력: {text}")
    print(f"예측된 감정: {', '.join(predicted_labels) if predicted_labels else '없음'}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L.U.N.A. Emotion Inference")
    parser.add_argument("--text", type=str, required=True, help="입력 텍스트")
    args = parser.parse_args()
    
    inter(args.text)