import torch
import numpy as np


def dice_coef_binary_class(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1e-4
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred):
    intersection = torch.sum(y_true == y_pred)
    smooth = torch.Tensor([1e-4]).to(intersection.device)
    dice = (2 * intersection + smooth) / (y_true.size(0) + y_pred.size(0) + smooth)

    return dice
