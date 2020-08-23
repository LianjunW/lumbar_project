import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MyBCEWithLogitsLoss,self).__init__()
        self.w = torch.tensor([1,2.4]).to(device)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.w)
    def forward(self, output_,target_):
        # targets = torch.zeros(targets.shape[0], self.cls_num).scatter_(1, targets, 1)

        cls_mask = torch.tensor([]).to(device)
        cls_mask = cls_mask.new_full(output_.shape, 0)
        ids = target_.view(-1,1).long()
        cls_mask.scatter_(1, ids.data, 1)
        return self.loss(output_,cls_mask)
        # return losses.mean()
class MyBCEWithLogitsLossSingleClass(nn.Module):
    def __init__(self):
        super(MyBCEWithLogitsLossSingleClass,self).__init__()
        self.w = torch.tensor([1.4]).to(device)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.w)
    def forward(self, output_,target_):
        target_ = target_.view(-1,1).to(device).float()
        return self.loss(output_,target_)
        # return losses.mean(