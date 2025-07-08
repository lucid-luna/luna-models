# models/emotion/emotion_model.py

import torch
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.config import load_config

tf_config = load_config("emotion_config")

class EmotionClassifier(torch.nn.Module):
    """
    L.U.N.A. Emotion Classifier Model
    """
    def __init__(self):
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
            # Multi-label binary cross-entropy loss
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )