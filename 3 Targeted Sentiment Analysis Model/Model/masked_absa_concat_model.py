import torch

from masked_absa_model import MaskedABSAModel


class MaskedABSAConcatModel(MaskedABSAModel):
    """
    TD-BERT equivalent where target and [CLS] outputs are being concatenated and then sent to the classification head
    """
    def __init__(self, checkpoint, num_labels=3, class_weights=None, cls_id=0, sep_id=0, mask_id=0):
        super(MaskedABSAConcatModel, self).__init__(checkpoint, num_labels, class_weights, 2,
                                                    cls_id=cls_id, sep_id=sep_id, mask_id=mask_id)


    def calculate_target_results(self, input_ids, base_outputs):
        """
        Calculates the input for the classification head of the model

        @param input_ids: encoded input sequence
        @param base_outputs: output tensor of the base model before the classification head
        @return: [CLS] output concatenated with the max-pooled outputs corresponding to the targets
        """
        targets = self.get_targets(input_ids, base_outputs)
        return torch.cat((targets, base_outputs.pooler_output), dim=1)
