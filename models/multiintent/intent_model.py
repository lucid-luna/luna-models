# models/intent/intent_model.py

import torch
from transformers import AutoModel
from utils.config import load_config

config = load_config("multiintent_config")

class MultiIntentClassifier(torch.nn.Module):
    """
    L.U.N.A. Multi-Intent Classifier Model
    """
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(self.dropout(pooled))
        return logits