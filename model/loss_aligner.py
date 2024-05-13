import torch
import torch.nn as nn

class AlignerLoss(nn.Module):
    """ Aligner Loss """

    def __init__(self, preprocess_config, model_config):
        super(AlignerLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

        self.repeat_num = model_config["aligner"]["repeat_num"]
        self.pho_to_con = model_config["aligner"]["pho_to_con"]
        

    def forward(self, inputs, predictions):
        (
            con_targets,
            _, # con_lens
            _, # max_con_len
            _, # duration_target 
        ) = inputs[-4:] 
        (
            con_predictions,
            src_masks,
            con_masks,
            _, # src_lens
            _, # con_lens
            tv_attn,
            diagonal_loss, 
        ) = predictions 

        src_masks = ~src_masks
        con_masks = ~con_masks
        
        con_targets.requires_grad = False
            
        if con_predictions.shape[1] > con_masks.shape[1]:
            con_predictions = con_predictions[:, :con_masks.shape[1], :] # trim tail
        else:
            con_masks = con_masks[:, :con_predictions.shape[1]] 
            con_targets = con_targets[:, :con_predictions.shape[1], :] if len(con_targets.shape) == 3 else con_targets[:, :con_predictions.shape[1]]

        con_predictions = con_predictions[con_masks]
        con_targets = con_targets[con_masks]
        con_loss = self.ce_loss(con_predictions, con_targets)
    
        total_loss = (con_loss + diagonal_loss)

        return (
            total_loss,
            con_loss,
            diagonal_loss,
        )
