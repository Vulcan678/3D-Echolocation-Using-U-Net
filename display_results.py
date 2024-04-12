from model import Unet1d_all
from utils import *
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from parallel_model_evaluate import seg_2_range, get_range_diff
import params
import NormalSignal


def main():
    np.set_printoptions(linewidth=np.inf)
    # set seed
    batch_size = 1
    seed_val = params.SEED
    seed_everything(seed_val)

    # set problem parameters
    number_of_targets = params.MAX_TARGETS
    dataset_folder = params.DATASET_FOLDER
    dataset_name = params.BASE_DATASET_NAME_3D
    data_sample_len = params.DATA_LEN
    label_len = params.LABEL_LEN
    torch.set_num_threads(params.MAX_CPUS)
    norm_ref = params.RX_SIGNAL_REF_GAIN
    norm_lambda = params.BOX_COX_LAMBDA

    norm_trans = NormalSignal.BoxCox(norm_ref, lamb=norm_lambda)
    train_set, val_set, test_set = create_dataset(dataset_folder, dataset_name, data_sample_len, label_len,
                                                           params, transform=norm_trans)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    # validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # get device
    device = get_device()
    print(device)

    kernel_size = params.KERNEL_SIZE
    max_ch = params.MAX_CH

    # loading the model
    # seg_model = Unet1d_all.U_Net(signal_ch=1, output_ch=1 + number_of_targets,
    #                              kernel_size=kernel_size, ch_max=max_ch).to(device)
    # seg_model = load_model(seg_model, "model_params/final/U_Net.pth", device)

    seg_model = Unet1d_all.AttU_Net(signal_ch=1, output_ch=1 + number_of_targets,
                                    kernel_size=kernel_size, ch_max=max_ch).to(device)
    seg_model = load_model(seg_model, "model_params/final/AttU_Net.pth", device)

    # n_head = params.N_TRANS_HEAD
    # drop_prob = params.DROP_PROB
    # n_trans_layers = params.N_TRANS_LAYERS
    # att_type = params.ATT_TYPE

    # seg_model = Unet1d_all.TransU_Net(signal_ch=1, output_ch=1 + number_of_targets,
    #                                   kernel_size=kernel_size, ch_max=max_ch,
    #                                   n_head=n_head, drop_prob=drop_prob,
    #                                   n_layers=n_trans_layers, att_type=att_type).to(device)
    # seg_model = load_model(seg_model, "model_params/final/TransU_Net.pth", device)

    display_samples(seg_model, test_loader, device)


# using the model for result display
def display_samples(seg_model, data_loader, device):
    threshold = params.SEG_2_RANGE_THRESHOLD
    min_range = params.DISTANCE_MIN
    max_range = params.DISTANCE_MAX
    mid_range = 0.5 * (min_range + max_range)
    range_range = 0.5 * (max_range - min_range)
    seg_model.eval()
    softmax = torch.nn.Softmax(dim=1)
    sample_num = 0
    range_diff_arr = np.array([])
    range_diff_gt_arr = np.array([])
    miss_detect_arr = np.array([])
    miss_detect_gt_arr = np.array([])

    # iterating over samples
    for data, ranges, seg_mask in data_loader:
        # pass the sample through the network
        data, seg_mask = data.to(device), seg_mask.to(device)

        print("sample {}".format(sample_num))

        padding = params.DATA_PADDING_LEN
        data = torch.nn.functional.pad(data, (0, padding), "constant", 0)
        seg_mask = torch.nn.functional.pad(seg_mask, (0, padding), "constant", 0)
        data, seg_mask = data.to(device), seg_mask.to(device)
        net_input_np = data.detach().cpu().numpy()
        net_input = data.unsqueeze(1)
        outputs = seg_model(net_input)
        outputs = softmax(outputs)
        seg_res = outputs.squeeze(1)
        gt_seg_mask = seg_mask.detach().cpu().numpy()
        seg_res = seg_res.detach().cpu().numpy()
        seg_estimation = np.argmax(seg_res, axis=1)

        # apply range estimation algorithm on true and estimated segmentation
        estimated_range_gt, range_index_gt, segs = seg_2_range(gt_seg_mask, threshold)
        estimated_range, range_index, segs = seg_2_range(seg_estimation, threshold)
        real_range = ranges.detach().cpu().numpy()

        estimated_range_gt = estimated_range_gt[np.abs(estimated_range_gt - mid_range) <= range_range]
        estimated_range = estimated_range[np.abs(estimated_range - mid_range) <= range_range]

        # calculate range estimation error from true segmentation (will be reference, close to zero)
        if np.size(estimated_range_gt) > 0:
            range_diff_gt = get_range_diff(real_range[0], estimated_range_gt)
            range_diff_gt1 = range_diff_gt[range_diff_gt < max_range - min_range]
            range_diff_gt_arr = np.append(range_diff_gt_arr, np.mean(range_diff_gt1))
        else:
            range_diff_gt1 = np.array([])

        # calculate range estimation error from true segmentation
        if np.size(estimated_range) > 0:
            range_diff = get_range_diff(real_range[0], estimated_range)
            range_diff1 = range_diff[range_diff < max_range - min_range]
            range_diff_arr = np.append(range_diff_arr, np.mean(range_diff1))
        else:
            range_diff1 = np.array([])

        real_range_wo_zeros = real_range[0]
        real_range_wo_zeros = real_range_wo_zeros[real_range_wo_zeros != 0]

        # calculating miss detect error
        miss_detect_gt_arr = np.append(miss_detect_gt_arr, np.size(real_range_wo_zeros) - np.size(range_diff_gt1))
        miss_detect_arr = np.append(miss_detect_arr, np.size(real_range_wo_zeros) - np.size(range_diff1))
        sample_num += 1

        number_of_targets = np.sum(real_range > 0)
        real_range = real_range[real_range > 0]

        range_diff_est = range_diff1

        print(range_diff_est)
        if np.size(range_diff_est) != np.size(real_range):
            print("target number missmatch")
            print("sample {}".format(sample_num))

        plt.figure()
        plt.plot(net_input_np[0])
        plt.title("number of targets {}".format(number_of_targets))
        plt.plot(seg_estimation[0])
        plt.plot(gt_seg_mask[0], color='red', linestyle='dotted')
        plt.show()

    # display range differance analysis
    range_diff_gt_arr_log10 = np.log10(range_diff_gt_arr)
    range_diff_arr_log10 = np.log10(range_diff_arr)
    cdf_prob_gt = np.array([i + 1 for i in range(len(range_diff_gt_arr))]) / len(range_diff_gt_arr)
    cdf_prob = np.array([i + 1 for i in range(len(range_diff_arr))]) / len(range_diff_arr)

    plt.figure()
    plt.hist(range_diff_gt_arr_log10, 10, facecolor='g')
    plt.xlabel('log10(mean_range_err_gt)')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(range_diff_arr_log10, 10, facecolor='g')
    plt.xlabel('log10(mean_range_err)')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(np.sort(range_diff_gt_arr_log10), cdf_prob_gt)
    plt.ylabel('CDF')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(-5.2, np.floor(np.max(range_diff_gt_arr_log10) / 0.2) * 0.2 + 0.2, 0.2))
    plt.xlabel('log10(mean_range_err_gt)')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(np.sort(range_diff_arr_log10), cdf_prob)
    plt.ylabel('CDF')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(-5.2, np.floor(np.max(range_diff_arr_log10) / 0.2) * 0.2 + 0.2, 0.2))
    plt.xlabel('log10(mean_range_err)')
    plt.grid(True)
    plt.show()

    plt.figure()
    n, bins, patches = plt.hist(miss_detect_gt_arr, int(np.max(miss_detect_gt_arr) - np.min(miss_detect_gt_arr)) + 1,
                                facecolor='g')
    plt.xlabel('miss_detect_gt')
    plt.grid(True)
    plt.show()
    print(n)

    plt.figure()
    n, bins, patches = plt.hist(miss_detect_arr, int(np.max(miss_detect_arr) - np.min(miss_detect_arr)) + 1,
                                facecolor='g')
    plt.xlabel('miss_detect')
    plt.grid(True)
    plt.show()
    print(n)


if __name__ == '__main__':
    main()
