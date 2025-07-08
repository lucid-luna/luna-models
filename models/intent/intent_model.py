# models/intent/intent_model.py

import torch
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.config import load_config

config = load_config("intent_config")

class IntentClassifier(torch.nn.Module):
    """
    L.U.N.A. Intent Classifier Model
    """
    def __init__(self):
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
        
    def forward(self, input_ids, attention_mask, labels=None):
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