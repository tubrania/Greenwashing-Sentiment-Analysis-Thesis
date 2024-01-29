import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class MaskedABSAModel(nn.Module):
    """
        Base class for the ABSA (Abstract Based Sentiment Analysis) problem
    """
    def __init__(self, checkpoint, num_labels=2, class_weights=None, hidden_size_multiplier=1, cls_id=0, sep_id=0, mask_id=0):
        super(MaskedABSAModel, self).__init__()

        self.base_model_config = AutoConfig.from_pretrained(checkpoint)
        self.base_model = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.5)
        self.num_labels = num_labels
        self.linear = nn.Linear(self.base_model_config.hidden_size * hidden_size_multiplier, self.num_labels)
        self.softmax = nn.Softmax()
        self.loss_function = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_id = mask_id

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs = self.calculate_target_results(input_ids, outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)
        logits = self.softmax(outputs)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

    def calculate_target_results(self, input_ids, base_outputs):
        return base_outputs.pooler_output


    def get_targets(self, input_ids, base_outputs) -> Tensor:
        """
        Finds the targets in the text, based on the second sequence after the separator token and returns
        base model outputs corresponding to the target tokens max pooled.
        @param input_ids: encoded input sequence
        @param base_outputs: output tensor of the base model before the classification head
        @return: tensor of outputs for the targets found in the sequence
        """
        targets = []
        for i, sequence in enumerate(input_ids):
            targets.append([])
            matches = torch.where(sequence == self.mask_id)
            targets[i] = torch.max(torch.cat([base_outputs.last_hidden_state[i][m] for m in matches], dim=1), dim=0).values

        targets = torch.cat([t.reshape(1, -1) for t in targets], dim=0)
        return targets
