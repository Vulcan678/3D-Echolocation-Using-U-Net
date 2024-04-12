import torch
import CNumpySonarDataSet
import os
import random
import numpy as np
from collections import OrderedDict


# check if "Cuda" (GPU) is available
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Cuda capability")
        print(torch.cuda.get_device_capability())
        print(torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    return device


# print model parameters
def print_model_params(model):
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of training parameters", pytorch_total_params)


# load datasets from numpy files
def create_dataset(folder_name, dataset_name, data_sample_len, label_len, params, transform=None,
                   target_transform=None):
    train_set = CNumpySonarDataSet.CNumpySonarDataSet(
        folder_name + "/train_" + dataset_name, data_sample_len, label_len, params, transform=transform,
        target_transform=target_transform)
    val_set = CNumpySonarDataSet.CNumpySonarDataSet(
        folder_name + "/val_" + dataset_name, data_sample_len, label_len, params, transform=transform,
        target_transform=target_transform)
    test_set = CNumpySonarDataSet.CNumpySonarDataSet(
        folder_name + "/test_" + dataset_name, data_sample_len, label_len, params, transform=transform,
        target_transform=target_transform)

    print("train db sample size ", train_set.__len__())
    print("validation db sample size ", val_set.__len__())
    print("test db sample size ", test_set.__len__())
    return train_set, val_set, test_set


# save model
def save_model(model, path, dist=False):
    with open(path, 'wb') as f:
        if dist:
            torch.save(model.module.state_dict(), f)
        else:
            torch.save(model.state_dict(), f)


# load model
def load_model(model, path, device):
    loaded_dict = torch.load(path, map_location=torch.device(device))
    fixed_dict = OrderedDict()
    for key in loaded_dict:
        if key.find("module.") == 0:
            fixed_dict[key[7:]] = loaded_dict[key]
        else:
            fixed_dict[key] = loaded_dict[key]

    model.load_state_dict(fixed_dict)
    return model


# set seed for all random processes
def seed_everything(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
