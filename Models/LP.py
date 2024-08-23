import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertForMultipleChoice


class CustomRobertaCLIPModel(nn.Module):
    def __init__(self,  num_classes):
        super(CustomRobertaCLIPModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.num_classes = num_classes

        # Freeze all layers of RoBERTa except the last attention layer
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze the last attention layer
        for param in self.roberta.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.project = nn.Linear(self.roberta.config.hidden_size + 768,
                                 self.roberta.config.hidden_size)

        for param in self.project.parameters():
            param.requires_grad = True

        # Classification head
        self.classifier = nn.Linear(self.roberta.config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, image_embeddings):
        # Get RoBERTa outputs up to the second-to-last layer
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-2]

        # Extend image embeddings to match the sequence length
        batch_size, seq_len, _ = hidden_states.size()
        image_embeddings = image_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate image embeddings with hidden states
        concatenated_states = torch.cat((hidden_states, image_embeddings), dim=-1)

        # Project the concatenated states back to the hidden size of RoBERTa
        projected_states = self.project(concatenated_states)

        # Forward pass through the modified last attention layer
        extended_attention_mask = self.roberta.get_extended_attention_mask(attention_mask, input_ids.shape,
                                                                           input_ids.device)
        attention_output = self.roberta.encoder.layer[-1](projected_states, extended_attention_mask)[0]

        # Mean pooling
        pooled_output = attention_output.mean(dim=1)

        # Classification
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)