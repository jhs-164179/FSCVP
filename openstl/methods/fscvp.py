import torch
from torch import nn
from openstl.models import FSCVP_Model
from .base_method import Base_method


class FSCVP(Base_method):
    def __init__(self, **args):
        super().__init__(**args)

        self.criterion2 = nn.L1Loss()

    def _build_model(self, **args):
        return FSCVP_Model(**args)
    
    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        # modify
        loss = (10 * self.criterion(pred_y, batch_y)) + self.criterion2(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss