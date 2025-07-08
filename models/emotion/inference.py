# model/emotion/inference.py

import torch
import argparse
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils.config import load_config
from models.emotion.emotion_model import EmotionClassifier

def infer(text: str, topk: int = 5):
    config = load_config("emotion_config")
    
    device = torch.device("cpu")
    
    # 1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    # 2) 모델 로드
    model = EmotionClassifier()
    model.load_state_dict(load_file(f"{config.train.output_dir}/model.safetensors"))
    model.to(device)
    model.eval()

    # 3) 입력 텍스트 토크나이징
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
    
    # 4) 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.sigmoid(logits)

        # ✅ Top-K 예측
        top_indices = torch.topk(probs, topk).indices.tolist()
        predicted_labels = [config.inference.label_list[i] for i in top_indices if i < len(config.inference.label_list)]
        
    # 5) 결과 출력
    print(f"📥 입력: {text}")
    print(f"🎯 Top-{topk} 예측 감정: {', '.join(predicted_labels) if predicted_labels else '없음'}")
    print(f"📊 확률 값: {probs[top_indices].tolist()}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L.U.N.A. Emotion Inference")
    parser.add_argument("--text", type=str, required=True, help="입력 텍스트")
    parser.add_argument("--topk", type=int, default=5, help="상위 K 예측 감정 수 (기본값: 5)")
    args = parser.parse_args()
    
    infer(args.text, topk=args.topk)