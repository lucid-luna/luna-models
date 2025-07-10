# ====================================================================
#  File: models/emotion/inference.py
# ====================================================================
"""
LunaEmotion 모델 추론 스크립트

학습된 모델을 사용하여 주어진 텍스트의 감정을 예측하고,
확률이 높은 상위 K개의 감정을 결과로 출력합니다.

실행 예시:
    python -m models.emotion.inference --text "<입력 텍스트>" --topk 3
"""

import torch
import argparse
from safetensors.torch import load_file
from transformers import AutoTokenizer
from typing import List, Tuple

from utils.config import load_config
from models.emotion.emotion_model import EmotionClassifier

# ----------
# CLI Entry Point
# ----------

def main():
    """
    CLI 진입점: 인자를 파싱하고, 추론 엔진을 실행하여 결과를 출력합니다.
    """
    parser = argparse.ArgumentParser(description="L.U.N.A 감정 분석 추론기")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="분석할 텍스트를 입력하세요."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="출력할 상위 K개 감정 수 (기본값: 3)"
    )
    args = parser.parse_args()

    engine = EmotionInference()
    predictions = engine.infer(text=args.text, top_k=args.topk)

    print("\n" + "=" * 40)
    print(f"입력 텍스트: \"{args.text}\"")
    print("─" * 40)
    print(f"Top-{args.topk} 예측 결과:")
    if not predictions:
        print("  - 예측된 감정이 없습니다.")
    else:
        for label, prob in predictions:
            print(f"  - {label}: {prob:.2%}")
    print("=" * 40)

class EmotionInference:
    """
    L.U.N.A Emotion Inference Engine

    속성:
        config: 모델 및 데이터 설정 로드 객체
        device: 'cuda' 또는 'cpu'
        tokenizer: 입력 텍스트 토크나이저
        model: EmotionClassifier 인스턴스
    """
    def __init__(self):
        self.config = load_config("emotion_config")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name, use_fast=True)
        
        self.model = EmotionClassifier()
        model_path = f"{self.config.train.output_dir}/model.safetensors"
        self.model.load_state_dict(load_file(model_path, device=self.device.type))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[L.U.N.A.] 모델 로딩 완료. 추론 장치: {self.device.type.upper()}")
        
    def infer(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        주어진 텍스트에 대해 감정 추론을 수행합니다.

        Args:
            text (str): 감정을 분석할 입력 텍스트
            top_k (int): 반환할 상위 예측 감정의 수

        Returns:
            List[Tuple[str, float]]: (감정 레이블, 확률) 튜플의 리스트
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.data.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.sigmoid(logits)

        top_probs, top_idxs = torch.topk(probs, top_k)
            
        label_list = self.config.inference.label_list
        results = []
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            if idx < len(label_list):
                results.append((label_list[idx], prob))
        return results
        
if __name__ == "__main__":
    main()