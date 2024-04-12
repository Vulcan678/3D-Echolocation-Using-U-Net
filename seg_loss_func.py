import torch.nn as nn
import torch.nn.functional


# verity of loss functions
class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-4

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)

        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


class DiceBCELoss(nn.Module):

    def __init__(self) -> None:
        super(DiceBCELoss, self).__init__()
        self.bce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        # compute softmax over the classes axis
        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        dice_loss = torch.mean(1. - dice_score)
        """
        dice_loss_func = DiceLoss()

        dice_loss = dice_loss_func(input, target)
        bce_loss = self.bce_loss(input, target)
        return dice_loss * 0.5 + bce_loss * 0.5


class IoULoss(nn.Module):

    def __init__(self) -> None:
        super(IoULoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        union = cardinality - intersection
        IoU = (intersection + self.eps) / (union + self.eps)
        return torch.mean(1. - IoU)


class FocalLoss(nn.Module):

    def __init__(self) -> None:
        super(FocalLoss, self).__init__()
        self.eps: float = 1e-6
        self.alpha: float = 0.8
        self.gamma: float = 2
        self.bce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_loss(input, target)
        bce_Exp = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - bce_Exp) ** self.gamma * bce_loss
        loss = torch.mean(focal_loss)
        return loss


class TverskyLoss(nn.Module):

    def __init__(self) -> None:
        super(TverskyLoss, self).__init__()
        self.eps: float = 1e-6
        # this loss function is weighted by the constants 'alpha' and 'beta' that penalise false positives and false
        # negatives respectively to a higher degree in the loss function as their value is increased larger βs weigh
        # recall higher than precision (by placing more emphasis on false negatives).

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        alpha: float = 0.2
        beta: float = 0.8

        input_soft = torch.nn.functional.softmax(input, dim=1)
        input_soft = torch.swapaxes(input_soft, 1, 2)
        # create the labels one hot tensor
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2)

        tp = torch.sum(input_soft * target_one_hot, dims)

        fp = torch.sum(input_soft * (1. - target_one_hot), dims)
        fn = torch.sum((1. - input_soft) * target_one_hot, dims)

        Tversky = tp / (tp + alpha * fp + beta * fn + self.eps)
        return torch.mean(1. - Tversky)
