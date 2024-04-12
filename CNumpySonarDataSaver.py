import numpy as np
from numpy.lib.format import open_memmap


# class for saving data samples, labels and masks
class NumpyCSonarDataSaver:
    def __init__(self, data_base_file_name, number_of_samples, data_sample_len, label_len):
        mask_len = data_sample_len
        self.numpy_mem_file = open_memmap(data_base_file_name, mode='w+', dtype=np.float,
                                          shape=(number_of_samples, data_sample_len + label_len + mask_len))
        self.data_counter = 0

    def add(self, waveform: np.ndarray, ranges: np.ndarray, seg_mask: np.ndarray):
        tmp = np.concatenate((waveform, ranges, seg_mask))
        self.numpy_mem_file[self.data_counter] = tmp
        self.data_counter += 1
