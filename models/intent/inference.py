# ====================================================================
#  File: models/intent/inference.py
# ====================================================================
"""
LunaIntent 모델 추론 스크립트

학습된 IntentClassifier 모델을 로드하여 입력 텍스트의 인텐트를 예측하고,
가장 높은 확률의 인텐트와 확률을 출력합니다.

실행 예시:
    python -m models.intent.inference --text "I'd like to cancel my subscription."
"""

import os
import argparse
from typing import Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from utils.config import load_config
from models.intent.intent_model import IntentClassifier

class IntentInference:
    """
    L.U.N.A Intent Inference Engine

    속성:
        config: 설정 객체
        device: 'cuda' 또는 'cpu'
        tokenizer: 입력 토큰화를 위한 토크나이저
        model: IntentClassifier 인스턴스
    """
    def __init__(self):
        self.config = load_config("intent_config")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            use_fast=True
        )

        self.model = IntentClassifier()
        model_path = os.path.join(self.config.train.output_dir, "model.safetensors")
        self.model.load_state_dict(load_file(model_path, device=self.device.type))
        self.model.to(self.device)
        self.model.eval()

        print(f"[L.U.N.A] Intent 모델 로딩 완료. 추론 장치: {self.device.type.upper()}")
    
    def infer(self, text: str) -> Tuple[str, float]:
        """
        입력 텍스트의 인텐트를 예측합니다.

        Args:
            text (str): 분석할 문장

        Returns:
            Tuple[str, float]: (예측 레이블, 확률)
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        pred_idx = torch.argmax(probs).item()
        label_list = self.config.inference.label_list
        label = label_list[pred_idx] if pred_idx < len(label_list) else "(unknown)"
        confidence = probs[pred_idx].item()

        return label, confidence

def main():
    parser = argparse.ArgumentParser(description="L.U.N.A Intent 추론기")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="추론할 입력 텍스트"
    )
    args = parser.parse_args()

    engine = IntentInference()
    label, conf = engine.infer(args.text)

    print("\n" + "=" * 40)
    print(f"입력 텍스트: \"{args.text}\"")
    print("─" * 40)
    print(f"예측된 인텐트: {label}")
    print(f"확률: {conf:.2%}")
    print("=" * 40)

if __name__ == "__main__":
    main()
