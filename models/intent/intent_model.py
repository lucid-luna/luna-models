# ====================================================================
#  File: models/intent/intent_model.py
# ====================================================================
"""
LunaIntent 모델 스크립트

Pretrained Transformer Encoder 기반 단일 레이블 인텐트 분류 모델
"""

import torch
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.config import load_config

config = load_config("intent_config")

# ----------
# Core Model Definition
# ----------

class IntentClassifier(torch.nn.Module):
    """
    L.U.N.A Intent Classifier

    Pretrained Transformer Encoder
    Dropout
    Linear Classification Head
    CrossEntropyLoss
    """
    def __init__(self):
        """
        모델 구성 요소 초기화

        Args:
            None
        """
        super().__init__()
        model_name = config.model.name
        num_labels = config.model.num_labels
        dropout_prob = getattr(config.model, "dropout_prob", 0.1)

        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=False
        )
        
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size,
            num_labels
        )
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        """
        Forward pass 수행

        Args:
            input_ids (torch.Tensor): 토큰 ID, shape (batch_size, seq_len)
            attention_mask (torch.Tensor): 어텐션 마스크, same shape as input_ids
            labels (torch.Tensor, optional): 정답 레이블, shape (batch_size,)

        Returns:
            SequenceClassifierOutput: loss 및 logits 포함
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0]
        
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )