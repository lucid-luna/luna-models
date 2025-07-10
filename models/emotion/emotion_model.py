# ====================================================================
#  File: models/emotion/emotion_model.py
# ====================================================================
"""
LunaEmotion 모델 스크립트
"""

import torch, os
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.config import load_config
from datasets import load_from_disk
import numpy as np

tf_config = load_config("emotion_config")

# ----------
# Helper functions
# ----------

def compute_pos_weight() -> torch.Tensor:
    """
    학습 데이터의 positive/negative 비율을 계산하여
    BCEWithLogitsLoss의 pos_weight로 사용할 텐서를 반환합니다.
    
    Returns:
        torch.Tensor: 각 레이블에 대한 계산된 pos_weight 텐서
    """
    train_ds = load_from_disk(os.path.join(tf_config.data.processed_dir, "train"))
    label_tensor = torch.stack([x["labels"] for x in train_ds])
    pos_counts = label_tensor.sum(dim=0)
    neg_counts = label_tensor.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)  # 0으로 나누는 것을 방지하기 위해 작은 값 추가
    return pos_weight

# ----------
# Core Model Definition
# ----------

class EmotionClassifier(torch.nn.Module):
    """
    Pretrained Transformer Encoder 다중 레이블 감정 분류 모델

    모델 구조:
    1. Pretrained Transformer Encoder
    2. Dropout Layer
    3. Linear Classifier Head
    """
    def __init__(self, pos_weight: torch.Tensor = None):
        """
        모델의 구성 요소를 초기화합니다.

        Args:
            pos_weight (torch.Tensor, optional): 손실 함수에 적용할 클래스 가중치
                                                None이면 내부적으로 `compute_pos_weight`를 호출하여 계산
        """
        super().__init__()
        model_name = tf_config.model.name
        num_labels = tf_config.model.num_labels
        
        dropout_prob = getattr(tf_config.model, "dropout_prob", 0.1)
        
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=False
        )
        
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size,
            num_labels
        )
        
        if pos_weight is None:
            pos_weight = compute_pos_weight()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        """
        모델의 forward pass를 수행합니다.

        Args:
            input_ids (torch.Tensor): 입력 토큰 ID 텐서
            attention_mask (torch.Tensor): 어텐션 마스크 텐서
            labels (torch.Tensor, optional): 다중 레이블 형식의 정답 텐서

        Returns:
            SequenceClassifierOutput: 손실(loss)과 로짓(logits)을 포함하는 객체
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # last_hidden_state의 shape: (batch_size, sequence_length, hidden_size)
        cls_emb = outputs.last_hidden_state[:, 0]
        
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )