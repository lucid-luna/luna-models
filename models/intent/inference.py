# model/intent/inference.py

import torch
import argparse
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils.config import load_config
from models.intent.intent_model import IntentClassifier

def infer(text: str):
    config = load_config("intent_config")
    device = torch.device("cpu")  # 또는 "cuda" 사용 가능
    
    # 1) 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    # 2) 모델 로드
    model = IntentClassifier()
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
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs).item()

    # 5) 라벨 매핑
    label_list = config.inference.label_list
    pred_label = label_list[pred_idx] if pred_idx < len(label_list) else "(unknown)"
    
    # 6) 결과 출력
    print(f"📥 입력: {text}")
    print(f"🎯 예측된 인텐트: {pred_label} (index: {pred_idx})")
    print(f"📊 확률 상위 인텐트: {probs[pred_idx].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L.U.N.A. Intent Inference")
    parser.add_argument("--text", type=str, required=True, help="입력 텍스트")
    args = parser.parse_args()

    infer(args.text)
