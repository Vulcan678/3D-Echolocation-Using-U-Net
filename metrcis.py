import numpy as np


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)
    corr = np.sum(SR == GT)
    tensor_size = SR.shape[0] * SR.shape[1]
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == np.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == True) + (GT == True)) == 2
    FN = ((SR == False) + (GT == True)) == 2

    SE = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) + (GT == 1)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    PC = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == np.max(GT)

    Inter = np.sum((SR + GT) == 2)
    Union = np.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == np.max(GT)

    Inter = np.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-4)

    return DC
