# ====================================================================
#  File: models/multiintent/inference.py
# ====================================================================
"""
LunaMultiIntent 모델 추론 스크립트

학습된 MultiIntentClassifier 모델을 로드하여 입력 텍스트의
상위 K개 인텐트를 확률과 함께 출력합니다.

실행 예시:
    python -m models.multiintent.inference --text "I'd like to cancel my subscription." --top_k 3
"""

import os
import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from utils.config import load_config
from models.multiintent.intent_model import MultiIntentClassifier

class MultiIntentInference:
    """
    L.U.N.A Multi-Intent 추론 클래스

    속성:
        config: 설정 객체
        device: 'cuda' 또는 'cpu'
        tokenizer: 입력 토큰화를 위한 토크나이저
        model: MultiIntentClassifier 인스턴스
    """
    def __init__(self):
        self.config = load_config("multiintent_config")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            use_fast=True
        )

        self.model = MultiIntentClassifier(
            model_name=self.config.model.name,
            num_labels=self.config.model.num_labels
        )
        model_path = os.path.join(self.config.train.output_dir, "model.safetensors")
        self.model.load_state_dict(load_file(model_path, device=self.device.type))
        self.model.to(self.device)
        self.model.eval()

        print(f"[L.U.N.A] Multi-Intent 모델 로딩 완료. 추론 장치: {self.device.type.upper()}")
        
    def infer(self, text: str, top_k: int = 2):
        """
        입력 텍스트에 대해 상위 K개 인텐트를 예측합니다.

        Args:
            text (str): 분석할 문장
            top_k (int): 반환할 상위 인텐트 개수

        Returns:
            List[Tuple[str, float]]: (레이블, 확률) 리스트
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
            "attention_mask": enc["attention_mask"].to(self.device)
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.sigmoid(logits)

        top_probs, top_idxs = torch.topk(probs, top_k)
        label_list = self.config.inference.label_list

        return [(label_list[idx], prob.item()) for idx, prob in zip(top_idxs.tolist(), top_probs)]


def main():
    parser = argparse.ArgumentParser(description="L.U.N.A Multi-Intent 추론기")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="추론할 입력 텍스트"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="출력할 상위 인텐트 개수 (기본값: 2)"
    )
    args = parser.parse_args()

    engine = MultiIntentInference()
    results = engine.infer(args.text, args.top_k)

    print("\n" + "=" * 40)
    print(f"입력 텍스트: \"{args.text}\"")
    print("─" * 40)
    print("예측된 인텐트:")
    for label, conf in results:
        print(f"  - {label:<30} ({conf:.2%})")
    print("=" * 40)

if __name__ == "__main__":
    main()
