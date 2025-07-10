# ====================================================================
#  File: models/multiintent/intent_model.py
# ====================================================================
"""
LunaMultiIntent 모델 정의

Pretrained Transformer Encoder 기반 다중 인텐트 분류 모델을 정의합니다.
    - Transformer Encoder
    - Linear Classification Head
    - BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
from transformers import AutoModel

# ----------
# Core Model Definition
# ----------

class MultiIntentClassifier(nn.Module):
    """
    L.U.N.A Multi-Intent Classifier

        1) Pretrained Transformer Encoder
        2) Linear Classification Head
        3) BCEWithLogitsLoss
    """
    def __init__(self, model_name: str, num_labels: int):
        """
        Args:
            model_name (str): Huggingface 사전 학습 모델 이름
            num_labels (int): 예측할 인텐트 수
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> dict:
        """
        Forward pass 수행

        Args:
            input_ids (torch.Tensor): 토큰 ID, shape=(batch_size, seq_len)
            attention_mask (torch.Tensor): 어텐션 마스크, same shape as input_ids
            token_type_ids (torch.Tensor, optional): 토큰 유형 ID
            labels (torch.Tensor, optional): 원-핫 인텐트 벡터, shape=(batch_size, num_labels)

        Returns:
            dict:
                "logits": torch.Tensor, shape=(batch_size, num_labels)
                "loss": torch.Tensor
        """
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_emb = encoder_outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(cls_emb)

        output = {"logits": logits}

        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels.float())

        return output
