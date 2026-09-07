# -*- coding: utf-8 -*-

import torch

from torch import nn
from torch.nn.functional import max_pool3d


class crossentry(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth))


class cross_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))

class dice_cross_loss(nn.Module):

    def __init__(self, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.smooth = 1e-6

    def forward(self, y_true, y_pred):
        # y_true shape : [N, 2, D, W, H]
        # y_pred shape : [N, 2, D, W, H]

        # dice loss
        y_pred_fg = y_pred[:, 1].reshape(-1)   # foreground probabilities
        y_true_fg = y_true[:, 1].reshape(-1)   # foreground labels
        intersection = (y_pred_fg * y_true_fg).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (y_pred_fg.sum() + y_true_fg.sum() + 1.0)

        # cross entropy loss
        ce_loss = -torch.mean(torch.sum(y_true * torch.log(y_pred + self.smooth), dim=1))

        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total_loss, self.lambda_dice*dice_loss, self.lambda_ce*ce_loss
    
class entropy_regularization_cross_loss(nn.Module):
    
    def __init__(self, beta_start=0.1, beta_end=0.001, decay_power=0.9, total_steps=10000, smooth=1e-6):
        super().__init__()
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.decay_power = decay_power
        self.total_steps = total_steps
        self.smooth      = smooth

    def get_beta(self, epoch):
        fraction = min(epoch / self.total_steps, 1.0)
        
        #return self.beta_start * (self.beta_end / self.beta_start) ** fraction # exponential decay
        return self.beta_end + (self.beta_start - self.beta_end) * (1 - fraction) ** self.decay_power # polynomial decay

    def forward(self, y_true, y_pred, epoch):
        beta = self.get_beta(epoch)

        ce_loss = -torch.mean(y_true * torch.log(y_pred + self.smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + self.smooth))
        
        clamped_pred = y_pred.clamp(min=1e-8)
        entropy = -(clamped_pred * clamped_pred.log()).sum(dim=-1).mean()

        return ce_loss - beta * entropy

class entropy_loss(nn.Module):

    def __init__(self, beta_start=0.1, beta_end=0.001, decay_power=0.9, total_steps=10000):
        super().__init__()
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.decay_power = decay_power
        self.total_steps = total_steps
        self.beta = None

    def get_beta(self, epoch):
        fraction = min(epoch / self.total_steps, 1.0)
        
        #return self.beta_start * (self.beta_end / self.beta_start) ** fraction # exponential decay
        return self.beta_end + (self.beta_start - self.beta_end) * (1 - fraction) ** self.decay_power # polynomial decay
    
    def forward(self, y_pred, epoch):
        self.beta = self.get_beta(epoch)
        clamped_pred = y_pred.clamp(min=1e-8)
        return -(clamped_pred * clamped_pred.log()).sum(dim=-1).mean() 

'''
Another Loss Function proposed by us in IEEE transactions on Image Precessing:
Paper: https://ieeexplore.ieee.org/abstract/document/9611074
Code: https://github.com/YaoleiQi/Examinee-Examiner-Network
'''


class Dropoutput_Layer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = (
            -((torch.sum(w * y_true * torch.log(y_pred + smooth)) /
               torch.sum(w * y_true + smooth)) +
              (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) /
               torch.sum(w * (1 - y_true) + smooth))) / 2)
        return loss_ce
