import numpy as np
from scipy import special


# log transformation
class NormalSignal:

    def __init__(self, ref_gain):
        self.ref_gain = ref_gain

    def __call__(self, sample):
        # scale sample with reference gain
        sample_norm = sample / self.ref_gain

        # perform log10 on positive on all values, keep sign and keep zeros as zeros
        ind_neg = np.where(sample_norm < 0)[0]
        ind_pos = np.where(sample_norm > 0)[0]
        ind_zeros = np.where(sample_norm == 0)[0]

        sample_norm1 = np.zeros_like(sample_norm)

        sample_norm1[ind_pos] = np.log10(sample_norm[ind_pos])
        sample_norm1[ind_neg] = -np.log10(-sample_norm[ind_neg])
        sample_norm1[ind_zeros] = 0

        return sample_norm1


# boxcox transformation, similar to log with more variability
class BoxCox:

    def __init__(self, ref_gain, lamb=0):
        self.ref_gain = ref_gain
        self.lamb = lamb

    def __call__(self, sample):
        # scale sample with reference gain
        sample_norm = sample / self.ref_gain

        # perform log10 on positive on all values, keep sign and keep zeros as zeros
        ind_neg = np.where(sample_norm < 0)[0]
        ind_pos = np.where(sample_norm > 0)[0]
        ind_zeros = np.where(sample_norm == 0)[0]

        sample_norm1 = np.copy(sample_norm)

        sample_norm1[ind_pos] = special.boxcox(sample_norm[ind_pos], self.lamb)
        sample_norm1[ind_neg] = -special.boxcox(-sample_norm[ind_neg], self.lamb)
        sample_norm1[ind_zeros] = 0

        return sample_norm1
