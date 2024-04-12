import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from numpy.lib.format import open_memmap


# customized dataset class for DataLoader
class CNumpySonarDataSet(Dataset):

    def __init__(self, data_base_file_name, data_sample_len, label_len, params,
                 transform=None, target_transform=None):
        self.numpy_mem_file = open_memmap(data_base_file_name, mode='r', dtype=np.float64)
        self.transform = transform
        self.target_transform = target_transform
        self.data_sample_len = data_sample_len
        self.seg_mask_len = self.data_sample_len
        self.label_len = label_len
        self.padding = params.DATA_PADDING_LEN_3D
        self.number_of_samples = int(
            self.numpy_mem_file.size / (self.label_len + self.data_sample_len + self.seg_mask_len))
        # pass
        max_range = 0
        min_range = params.DISTANCE_MAX+1

        # get max and min range
        for index in range(0, self.number_of_samples):
            data_with_label = self.numpy_mem_file[index]
            num_target = np.sum(
                data_with_label[self.data_sample_len:self.data_sample_len + self.label_len // 3] > 0)
            range_list = data_with_label[self.data_sample_len:
                                         self.data_sample_len + num_target]
            if np.max(range_list) > max_range:
                max_range = np.max(range_list)
                self.max_waveform_index = index

            if np.min(range_list) < min_range:
                min_range = np.min(range_list)
                self.min_waveform_index = index

        self.ref_gain = params.RX_SIGNAL_REF_GAIN

    def __getitem__(self, index):

        data_with_label = self.numpy_mem_file[index]
        data = np.copy(data_with_label[0:self.data_sample_len])
        label = np.copy(
            data_with_label[self.data_sample_len:self.data_sample_len + self.label_len])
        seg_mask1 = np.copy(data_with_label[self.data_sample_len + self.label_len:])

        if self.transform is not None:
            data = self.transform(data)

        # make data and segmentation torch compatible
        data = torch.from_numpy(data).float()
        seg_mask1 = torch.from_numpy(seg_mask1).float()

        # if transform is needed perform padding to fit convolution down sampling structure
        # perform for each ear (input channel) separately
        if self.transform is not None:
            data_left = data[:self.data_sample_len // 2]
            data_right = data[self.data_sample_len // 2:]
            data_left = torch.nn.functional.pad(data_left, (0, self.padding),
                                                "constant", 0)
            data_right = torch.nn.functional.pad(data_right, (0, self.padding),
                                                 "constant", 0)
            data = torch.concat((data_left, data_right))
            seg_mask1_left = seg_mask1[:self.data_sample_len // 2]
            seg_mask1_right = seg_mask1[self.data_sample_len // 2:]
            seg_mask1_left = torch.nn.functional.pad(seg_mask1_left, (0, self.padding),
                                                     "constant", 0)
            seg_mask1_right = torch.nn.functional.pad(seg_mask1_right, (0, self.padding),
                                                      "constant", 0)
            seg_mask1 = torch.concat((seg_mask1_left, seg_mask1_right))

        label = torch.from_numpy(label)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label, seg_mask1

    def __len__(self):
        return self.number_of_samples

    def get_max_range_sample(self):
        return self.max_waveform_index

    def get_min_range_sample(self):
        return self.min_waveform_index

    def get_ref(self):
        return self.ref_gain
