# models/emotion/emotion_model.py

import torch, os
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.config import load_config
from datasets import load_from_disk
import numpy as np

tf_config = load_config("emotion_config")

def compute_pos_weight():
    """
    Positive weight 계산 함수
    """
    train_ds = load_from_disk(os.path.join(tf_config.data.processed_dir, "train"))
    label_tensor = torch.stack([x["labels"] for x in train_ds])
    pos_counts = label_tensor.sum(dim=0)
    neg_counts = label_tensor.shape[0] - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)  # 0으로 나누는 것을 방지하기 위해 작은 값 추가
    return pos_weight

class EmotionClassifier(torch.nn.Module):
    """
    L.U.N.A. Emotion Classifier Model
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        model_name = tf_config.model.name
        num_labels = tf_config.model.num_labels
        
        # Optional Dropout (기본값 : 0.1)
        dropout_prob = getattr(tf_config.model, "dropout_prob", 0.1)
        
        # 1) Pretrained Encoder 로드
        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=False
        )
        
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        # 2) Dropout + Classification Head
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size,
            num_labels
        )
        
        pos_weight = compute_pos_weight()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else None
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 3) Encoder 출력
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 4) [CLS] 토큰 임베드 추출
        cls_emb = outputs.last_hidden_state[:, 0]
        
        # 5) Dropout -> Linear -> Sigmoid
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        
        # 6) Loss 계산 (Multi-label classification)
        loss = None
        
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )