import time
import pandas as pd
from model import Unet1d_all
from utils import *
from torch.utils.data import DataLoader
import collections
import sys
import glob
import os
import params
import NormalSignal
from multiprocessing import Pool
import CReflectorsGenerator
import gc


def main(args):
    # setting parameters for evaluation
    params.SEG_2_RANGE_THRESHOLD = 50
    seg_2_range_th = params.SEG_2_RANGE_THRESHOLD

    dataset_folder = params.DATASET_FOLDER
    dataset_name = params.BASE_DATASET_NAME_3D
    data_sample_len = 2 * params.DATA_LEN_3D
    label_len = params.LABEL_LEN_3D
    norm_ref = params.RX_SIGNAL_REF_GAIN

    torch.set_num_threads(params.MAX_CPUS)

    # set seed
    seed_val = params.SEED
    seed_everything(seed=seed_val)

    # load dataset with transformation
    norm_trans = NormalSignal.BoxCox(norm_ref, lamb=params.BOX_COX_LAMBDA)
    train_set, val_set, test_set = create_dataset(dataset_folder, dataset_name, data_sample_len, label_len,
                                                           params, transform=norm_trans)

    # get device
    device = params.DEVICE

    model_params_path = args[1]
    print_plots = args[2]

    sub_locs = [i for i in range(0, len(model_params_path))
                if model_params_path[i:].startswith("_")]

    eval_pathes = [model_params_path]
    test_runs = np.array([0])

    # in case of updating summary after the training finished
    if len(args) == 4:
        print("long run")
        summery_path = args[3]

        # load summary csv
        df = pd.read_csv(summery_path)

        # filter only runs good initial performance and that wasn't already evaluated
        df_fit = df.loc[df["epoch stop"] > df["num epochs"] / 2]
        df_fit = df_fit.loc[np.log10(1 - df_fit['dice']) <= np.log10(1 - df_fit['dice']).mean()]
        df_fit = df_fit.loc[df_fit['Avg FA rate'] >= 1]
        df_fit = df_fit.loc[df_fit['dice'] >= 0.95]
        test_runs = np.array(df_fit.loc[:, 'run_num'])

        # prepare for iterative evaluation of models
        date_str = model_params_path[sub_locs[-7] + 1:sub_locs[-1]]
        eval_pathes = glob.glob(params.PARAMS_FOLDER + "/" + date_str[:-5] + '/end_*_' + date_str + '_*.pth')
        eval_pathes.sort(key=os.path.getmtime)

        if len(eval_pathes) == 0:
            eval_pathes = [model_params_path]
            test_runs = np.array([0])

    eval_res_ar = []
    str_ar = []
    # perform performance evaluation for each model
    for run_num in test_runs:
        path = eval_pathes[run_num]
        print(run_num)
        print(path)

        # start evaluation timing
        start_time = time.time()

        # perform evaluation
        avg_err, avg_false_target_rate, miss_detection_rate, false_target_dict, miss_detect_dict, target_counters = evaluate_model(
            path, val_set, device, seg_2_range_th, print_plots)
        run_time = time.time() - start_time

        # print evaluation performance summary
        str1 = "mse {0:.2e} ,false target {1:.2e} miss detected {2:.2e} \n".format(
            avg_err, avg_false_target_rate, miss_detection_rate)

        str1 = str1 + "false target statistic\n"
        for number_of_targets in false_target_dict:
            number_of_false_targets_per_target = len(false_target_dict[number_of_targets])
            number_of_samples_per_target = target_counters[number_of_targets]
            err_percent = number_of_false_targets_per_target / number_of_samples_per_target * 100
            str1 = str1 + "for {} targets \n".format(number_of_targets)
            str1 = str1 + "\t number of false targets {0:d} out of {1:d} {2:.2f} %\n".format(
                number_of_false_targets_per_target, number_of_samples_per_target, err_percent)
            str1 = str1 + "\t difference from real number of targets\n"
            counter = collections.Counter(false_target_dict[number_of_targets])
            str1 = str1 + '\t' + str(counter) + "\n"

        str1 = str1 + "miss detect statistic\n"
        for number_of_targets in miss_detect_dict:
            number_of_miss_detect_targets_per_target = len(
                miss_detect_dict[number_of_targets])
            number_of_samples_per_target = target_counters[number_of_targets]
            err_percent = number_of_miss_detect_targets_per_target / number_of_samples_per_target * 100

            str1 = str1 + "for {} targets \n".format(number_of_targets)
            str1 = str1 + "\t number of miss detect targets {0:d} out of {1:d} {2:.2f} %\n".format(
                number_of_miss_detect_targets_per_target, number_of_samples_per_target, err_percent)
            str1 = str1 + "\t difference from real number of targets\n"
            counter = collections.Counter(miss_detect_dict[number_of_targets])
            str1 = str1 + '\t' + str(counter) + "\n"

        str_ar.append(str1)

        eval_res = [avg_err, avg_false_target_rate, miss_detection_rate]
        eval_res_ar.append(eval_res)

        # update summary CSV with performance results
        if len(args) == 4:
            df.loc[df['run_num'] == run_num, ['Avg Error', 'Avg FA rate', 'Avg MD rate', 'eval run time']] = eval_res

            df.to_csv(summery_path, index=False)

    return str_ar, eval_res_ar, eval_pathes, run_time


def evaluate_model(path, test_set, device, seg_2_range_th=10, print_plots=False):
    # initialize performance evaluation parameters
    total_real_targets = 0
    range_err_sum = 0
    false_target_sum = 0
    miss_detect_sum = 0
    number_of_classes = params.MAX_TARGETS + 1

    false_target_dict = {}
    miss_detect_dict = {}
    target_counters = {}
    for i in range(1, params.MAX_TARGETS + 1):
        false_target_dict[i] = []
        miss_detect_dict[i] = []
        target_counters[i] = 0

    # from path name find model hyperparameters
    sub_locs_path = [i for i in range(0, len(path)) if path[i:].startswith("_")]

    model_name = path[sub_locs_path[8] + 1:sub_locs_path[10]]
    kernel_size = int(path[sub_locs_path[17] + 1:sub_locs_path[18]])
    max_ch = int(path[sub_locs_path[18] + 1:sub_locs_path[19]])

    # load your model
    if model_name == "U_Net":
        seg_model = Unet1d_all.U_Net(signal_ch=1, output_ch=number_of_classes,
                                     kernel_size=kernel_size, ch_max=max_ch)
    elif model_name == "AttU_Net":
        seg_model = Unet1d_all.AttU_Net(signal_ch=1, output_ch=number_of_classes,
                                        kernel_size=kernel_size, ch_max=max_ch)
    elif model_name == "R2AttU_Net":
        seg_model = Unet1d_all.R2AttU_Net(signal_ch=1, output_ch=number_of_classes,
                                          kernel_size=kernel_size, ch_max=max_ch)
    elif model_name.find("Trans") >= 0:
        # from path name find model hyperparameters
        n_head = int(path[sub_locs_path[21] + 1:sub_locs_path[22]])
        drop_prob = 0
        n_trans_layers = int(path[sub_locs_path[23] + 1:sub_locs_path[24]])
        att_type = path[sub_locs_path[24] + 1:sub_locs_path[25]]
        if model_name == "TransU_Net":
            seg_model = Unet1d_all.TransU_Net(signal_ch=1, output_ch=number_of_classes,
                                              kernel_size=kernel_size, ch_max=max_ch,
                                              n_head=n_head, drop_prob=drop_prob,
                                              n_layers=n_trans_layers, att_type=att_type)
        elif model_name == "ClassicTransU_Net":
            model = Unet1d_all.ClassicTransU_Net(signal_ch=1, output_ch=number_of_classes,
                                                 kernel_size=kernel_size, ch_max=max_ch,
                                                 n_head=n_head, drop_prob=drop_prob,
                                                 n_layers=n_trans_layers)
        elif model_name == "R2TransU_Net":
            seg_model = Unet1d_all.R2TransU_Net(signal_ch=1, output_ch=number_of_classes,
                                                kernel_size=kernel_size, ch_max=max_ch,
                                                n_head=n_head, drop_prob=drop_prob,
                                                n_layers=n_trans_layers, att_type=att_type)

    # load model and enter evaluation mode
    seg_model = seg_model.to(device)
    seg_model = load_model(seg_model, path, device)
    seg_model.eval()

    # find the largest batch size to perform evaluation faster without memory overflow
    den = 50
    batch_size = len(test_set) // den
    was_error = True
    while was_error and batch_size >= 1:
        try:
            print("batch_size: " + str(batch_size))
            values = []
            count = 0

            # initialize DataLoader and iterate over batches
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            for data, ranges, seg_mask in test_loader:
                data_ar_ears = [data[:, :params.DATA_LEN_ROUNDED_3D], data[:, params.DATA_LEN_ROUNDED_3D:]]

                # get model output segmentation and true ranges for each ear
                left = True
                for data1 in data_ar_ears:
                    # pass Rx signal through the network
                    seg_model.eval()
                    data1 = data1.to(device)
                    net_input = data1.unsqueeze(1)
                    outputs = seg_model(net_input)

                    seg_res = outputs.squeeze(1)
                    seg_res_np_ar = seg_res.detach().cpu().numpy()
                    range_np_ar = ranges.detach().cpu().numpy()

                    # get true range for each ear
                    for seg_res_np, range_np in zip(seg_res_np_ar, range_np_ar):
                        range_np1 = range_np[:params.MAX_TARGETS]
                        num_targets = np.sum(range_np1 > 0)

                        # for each target get range to relevant ear
                        for ind_target in range(num_targets):
                            Dist = range_np[ind_target]
                            theta = range_np[ind_target + params.MAX_TARGETS]
                            phi = range_np[ind_target + 2 * params.MAX_TARGETS]

                            range_ears_ar, _ = CReflectorsGenerator.target_loc_to_2ears(
                                Dist, theta, phi)

                            if left:
                                range_np1[ind_target] = range_ears_ar[0]
                            else:
                                range_np1[ind_target] = range_ears_ar[1]

                        # save segmentation output and true range
                        inputs = (seg_res_np, seg_2_range_th, range_np1, count)
                        values.append(inputs)
                        count += 1

                    left = False

            was_error = False
        except:
            # if current batch size hit memory overflow make batch size smaller
            del values
            gc.collect()
            den += 1
            while np.mod(len(test_set), den) != 0:
                den += 1
            batch_size = len(test_set) // den

    batch_size = max(batch_size, 1)
    print("batch_size: " + str(batch_size))
    # delete model to free the memory
    del seg_model

    # perform parallel evaluation of batches on different CPU's
    print(params.MAX_CPUS)
    with Pool(params.MAX_CPUS) as p:
        # pass values array with data for evaluation
        results = p.map(singel_evaluate, values)

        # analyze the evaluation results to get total performance evaluation
        for res in results:
            number_of_targets = res[0]
            target_diff = res[1]
            range_error = res[2]
            false_target = res[3]
            miss_detect = res[4]

            total_real_targets += number_of_targets
            target_counters[number_of_targets] += 1

            if target_diff == 0:
                range_err_sum += range_error
            elif target_diff > 0:
                false_target_sum += false_target
                false_target_dict[number_of_targets].append(target_diff)
            else:
                miss_detect_sum += miss_detect
                miss_detect_dict[number_of_targets].append(-target_diff)

        p.close()  # prevent new tasks from being submitted
        p.join()  # wait for all worker processes to finish
        p.terminate()  # terminate any remaining worker processes

    # free the memory
    del values
    gc.collect()

    # calculate final performance metrics
    avg_err = range_err_sum / total_real_targets
    false_target_rate = false_target_sum / total_real_targets
    miss_detect_rate = miss_detect_sum / total_real_targets
    return avg_err, false_target_rate, miss_detect_rate, false_target_dict, miss_detect_dict, target_counters


# perform performance evaluation on a single batch
def singel_evaluate(inputs):
    seg_res_np = inputs[0]
    seg_2_range_th = inputs[1]
    range_np = inputs[2]

    # print progress
    if np.mod(inputs[3], 500) == 0:
        print(inputs[3])

    max_range = params.DISTANCE_MAX + params.DIST_EARS / 2
    min_range = params.DISTANCE_MIN - params.DIST_EARS / 2

    seg_label = np.argmax(seg_res_np, axis=0)

    # range estimation and segmentation start and end positions from segmentation
    range_estimation, segs = seg_2_range(seg_label, threshold=seg_2_range_th)
    # handle cases of false detection when we detect more targets or fewer targets
    number_of_targets = np.sum(range_np[:params.MAX_TARGETS] > 0)
    # remove zeroes
    real_range = range_np[range_np[:params.MAX_TARGETS] > 0]

    # calculate detected targets difference and range differences
    target_diff = np.size(range_estimation) - number_of_targets
    range_diff = get_range_diff(real_range, range_estimation)
    range_diff = range_diff[range_diff <= max_range - min_range]

    # calculate batch performance metrics
    range_error = 0
    false_target = 0
    miss_detect = 0
    if target_diff == 0:
        range_error = np.sum(np.abs(range_diff))
    elif target_diff > 0:
        false_target = np.size(range_estimation) - np.size(range_diff)
    else:
        miss_detect = number_of_targets - np.size(range_diff)

    return number_of_targets, target_diff, range_error, false_target, miss_detect


# estimate from segmentation targets range based on Tx length and error threshold
def seg_2_range(seg_results: np.ndarray, threshold=10):
    range_min_dist = 10 ** (-params.NUM_DECIMALS_ROUND)
    range_min_dist_in_ind = range_min_dist * params.DISTANCE2IND

    max_range = params.DISTANCE_MAX + params.DIST_EARS / 2
    min_range = params.DISTANCE_MIN - params.DIST_EARS / 2
    ind_in_range = params.IND2DISTANCE

    threshold = np.max([threshold, range_min_dist_in_ind])

    seg_results = np.round(seg_results).astype(int)

    # calculate jumps and drops in segmentation, jumps as target position, drops as end of received echo
    delta_seg_results = seg_results[1:] - seg_results[: -1]

    # split between up and down change in segmentation
    up_seg = np.array([])
    down_seg = np.array([])

    for i in range(len(delta_seg_results)):
        dseg = delta_seg_results[i]
        if dseg > 0:
            up_seg = np.append(up_seg, i * np.ones(dseg))
        if dseg < 0:
            down_seg = np.append(down_seg, i * np.ones(-dseg))

    # get valid segments from each detected target
    segs = get_ind_segs(up_seg, down_seg, threshold, seg_results)

    # calculate locations of the first echo
    if np.size(segs) == 0:
        echo_locations = 2 * max_range * params.DISTANCE2IND
    else:
        if np.size(segs) == 2:
            segs = np.reshape(segs, (-1, 2))

        # calculate echo locations as mean between start of the echo and (end of the echo - echo length)
        start_echo_locations = np.squeeze(segs[:, 0])
        end_echo_locations = np.squeeze(segs[:, 1])
        echo_locations = (start_echo_locations + end_echo_locations - params.TX_SIGNAL_LENGTH_IND).astype(float) / 2

        if len(np.shape(echo_locations)) == 0:
            echo_locations = np.array([echo_locations])
            start_echo_locations = np.array([start_echo_locations])
            end_echo_locations = np.array([end_echo_locations])

        # filter and clip echo start and end locations by maximal and minimal range
        for i in range(len(echo_locations)):
            if start_echo_locations[i] > int(min_range * params.DISTANCE2IND) - 3 and end_echo_locations[i] < int(
                    max_range * params.DISTANCE2IND) - 1:
                echo_locations[i] = np.clip(echo_locations[i], int(min_range * params.DISTANCE2IND) - 2,
                                            int(max_range * params.DISTANCE2IND) - 2)

    # calculate ranges from echo locations
    ranges = (echo_locations + 2) * params.IND2DISTANCE
    if len(np.shape(ranges)) == 0:
        ranges = np.array([ranges])
    else:
        ranges = np.sort(ranges)

    # filter and clip ranges out of valid values
    ranges = ranges[ranges + ind_in_range >= min_range]
    ranges = ranges[ranges - ind_in_range <= max_range]
    ranges = np.clip(ranges, min_range, max_range)

    return ranges, segs


# filter true segments by up to down distance matching to Tx length (with error threshold)
def get_ind_segs(up_inds, down_inds, threshold, seg_results):
    tx_signal_len = params.TX_SIGNAL_LENGTH_IND
    max_ind = params.DATA_LEN_ROUNDED_3D
    up_inds = np.array(up_inds)
    down_inds = np.array(down_inds)

    # fix empty up\down indices
    if len(down_inds) == 0:
        down_inds = np.array([max_ind])

    if len(up_inds) == 0:
        up_inds = np.array([0])

    seg_results = np.copy(seg_results)

    # filter non logical down indices
    if down_inds[0] - up_inds[0] < tx_signal_len - 3 * threshold:
        ind_keep = np.where(down_inds - up_inds[0] >= tx_signal_len - 3 * threshold)[0]
        down_inds = down_inds[ind_keep]

    # in case of no down indices return default
    if len(down_inds) == 0:
        segs = np.array([10 * max_ind, 50 * max_ind])
        return segs

    # filter non logical up indices
    if down_inds[-1] - up_inds[-1] < tx_signal_len - 3 * threshold:
        ind_keep = np.where(down_inds[-1] - up_inds >= tx_signal_len - 3 * threshold)[0]
        up_inds = up_inds[ind_keep]

    # in case of no up indices return default
    if len(up_inds) == 0:
        segs = np.array([10 * max_ind, 50 * max_ind])
        return segs

    # iterative up to down index matching based on minimal error from Tx length
    count = 0
    segs = np.array([10 * max_ind, 50 * max_ind])
    while len(down_inds) > 0 and len(up_inds) > 0:
        # build for each pair of up and down index the minimal difference
        # with possibility to account for multiple consecutive Rx echos that merge to one large echo
        diff_mat, n_diff_mat = get_segs_for_min_err(up_inds[0], down_inds, threshold)

        for i in range(1, len(up_inds)):
            diff_ar, n_diff = get_segs_for_min_err(up_inds[i], down_inds, threshold)
            diff_mat = np.vstack((diff_mat, diff_ar))
            n_diff_mat = np.vstack((n_diff_mat, n_diff))

        # for each up index find the best down index
        # saving the error and number of potential echos from up to down index
        if len(np.shape(diff_mat)) == 2:
            min_diff_arr = np.min(diff_mat, axis=1)
            ind_min_diff_arr = np.argmin(diff_mat, axis=1)
            min_n_diff_arr = [n_diff_mat[k, ind_min_diff_arr[k]] for k in range(len(ind_min_diff_arr))]
        else:
            min_diff_arr = np.array([np.min(diff_mat)])
            ind_min_diff_arr = np.array([np.argmin(diff_mat)])
            min_n_diff_arr = np.array([n_diff_mat[ind_min_diff_arr]])

        # round indexes to int
        ind_min_diff_arr = ind_min_diff_arr.astype(int)
        min_n_diff_arr = np.array(min_n_diff_arr).astype(int)

        # in case of unique pairs build the segs matrix with up and down indexes of each echo (or consecutive echos)
        if len(np.unique(ind_min_diff_arr)) == len(ind_min_diff_arr):
            valid_resualts_inds = np.where(min_diff_arr <= threshold)[0]
            up_inds_used = np.copy(up_inds[valid_resualts_inds])
            down_inds_used = np.copy(down_inds[ind_min_diff_arr[valid_resualts_inds]])

            up_inds = np.delete(up_inds, valid_resualts_inds)
            down_inds = np.delete(down_inds, ind_min_diff_arr[valid_resualts_inds])

            segs_temp = np.transpose(np.vstack((up_inds_used, down_inds_used)))
            segs = np.vstack((segs, segs_temp))
            break

        if np.min(min_diff_arr) > threshold:
            # break if all the pairs are with error larger than the threshold
            break
        else:
            # find pairs with minimal potential echos between up and down indices with echo length in length threshold
            n = np.max([np.min(min_n_diff_arr), 1]) - 1
            found = False

            while not found:
                n = n + 1
                ind_up_n = np.where(min_n_diff_arr == n)[0]
                if len(ind_up_n) > 0:
                    min_diff = np.min(min_diff_arr[ind_up_n])
                    if min_diff <= threshold:
                        found = True
                        ind_up_min = ind_up_n[np.argmin(min_diff_arr[ind_up_n])]

            # add up and down indices with the lowest echo length error
            ind_down_min = ind_min_diff_arr[ind_up_min]
            segs_temp = np.array([up_inds[ind_up_min], down_inds[ind_down_min]])

            if count == 0:
                segs = np.copy(segs_temp)
            else:
                segs = np.vstack((segs, segs_temp))

            # delete from up and down arrays the used indices
            up_inds = np.delete(up_inds, ind_up_min)
            down_inds = np.delete(down_inds, ind_down_min)
            count += 1

    # fix segs matrix to required shape and index limit
    if np.size(segs) == 2:
        segs = np.reshape(segs, (1, 2))

    if np.size(segs) >= 2:
        ind_off = np.where(segs[:, 0] > max_ind + 2)[0]
        segs = np.delete(segs, ind_off, axis=0)

    # take the found initial guess and fit it to minimize segmentation error from network output

    # save unused indices in fitting name
    up_inds_unused = up_inds
    down_inds_unused = down_inds

    # perform the minimization of segmentation error for initial guess
    results = fix_segs_min_err_from_res(segs, seg_results, up_inds_unused, down_inds_unused, threshold)

    segs, up_inds_unused, down_inds_unused = results

    # find up and down indices of multiple echos and add them to segs array
    for i in range(len(segs[:, 0])):
        diff = float(segs[i, 1] - segs[i, 0])
        if diff >= 1.5 * tx_signal_len:
            n_segs_in_seg = int(np.round(diff / tx_signal_len))
            seg_size = diff / n_segs_in_seg
            segs[i, 1] = segs[i, 0] + seg_size
            for k in range(1, n_segs_in_seg):
                up_ind1 = segs[i, 0] + k * seg_size
                down_ind1 = segs[i, 0] + (k + 1) * seg_size
                segs = np.vstack((segs, np.array([up_ind1, down_ind1])))

    # sort segs matrix by up index
    if np.size(segs) > 2:
        segs = segs[segs[:, 0].argsort()]

    return segs


# fix the initial guess of segs to achieve segmentation with minimal differance from segmentation result of the network
def fix_segs_min_err_from_res(segs, seg_results, up_inds_unused, down_inds_unused, threshold):
    tx_signal_len = params.TX_SIGNAL_LENGTH
    max_ind = params.DATA_LEN_ROUNDED

    # start function timing
    start_time = time.time()

    # create array to store helping data
    memory_ar = []
    res_score = np.array([])
    finish_status = np.array([])
    min_res_score = max_ind ** 2

    # limit function run time to 20 seconds (in case of many noises)
    while time.time() - start_time < 20:
        if len(memory_ar) == 0:
            # on first run perform copying of arrays and saving the first attempt
            segs_temp = np.copy(segs)  # segs matrix with up and down indices of each echo
            up_inds_unused_temp = np.copy(up_inds_unused)  # unused in segs up indices got from network output
            down_inds_unused_temp = np.copy(down_inds_unused)  # unused in segs down indices got from network output
            add_to_miss_th = 0  # increment value to detection threshold
            ignore_FA = 0  # index up to which we should ignore FA events
            ignore_MD = 0  # index up to which we should ignore MD events

            memory = {}
            memory["segs"] = np.copy(segs_temp)
            memory["up_inds_unused"] = np.copy(up_inds_unused_temp)
            memory["down_inds_unused"] = np.copy(down_inds_unused_temp)
            memory["add"] = add_to_miss_th
            memory["ignore_FA"] = ignore_FA
            memory["ignore_MD"] = ignore_MD

            memory_ar.append(memory)
            i = 0
        elif len(np.where(finish_status == 0)[0]) > 0:
            # in case of unfinished evaluation of cases load case parameters from memory
            i = np.where(finish_status == 0)[0][0]
            memory = memory_ar[i]
            segs_temp = memory["segs"]
            up_inds_unused_temp = memory["up_inds_unused"]
            down_inds_unused_temp = memory["down_inds_unused"]
            add_to_miss_th = memory["add"]
            ignore_FA = memory["ignore_FA"]
            ignore_MD = memory["ignore_MD"]
        else:
            break

        # get segmentation from segs matrix
        seg_after_alg = seg_mat2segmentation(segs_temp)

        # calculate differance for each sample between network output and segmentation from segs
        diff = seg_results - seg_after_alg

        # mark indices where the diff is non-zero
        diff_FA = np.zeros_like(diff)
        diff_MD = np.zeros_like(diff)
        diff_FA[diff < 0] = -1
        diff_MD[diff > 0] = 1

        # perform cumulative sum for each array to distinguish between noise and continues differance
        ret_FA = np.cumsum(diff_FA)
        ret_MD = np.cumsum(diff_MD)

        # make the sum only on n last values
        n = int(max(tx_signal_len // 3, 3 * threshold))
        ret_FA[n:] = ret_FA[n:] - ret_FA[:-n]
        ret_MD[n:] = ret_MD[n:] - ret_MD[:-n]

        # save in score the calculated sum of FA and MD drifts and update finish_status to 1
        if i == 0:
            res_score = np.append(res_score, np.sum(np.abs(ret_FA)) + np.sum(np.abs(ret_MD)))
            finish_status = np.append(finish_status, 1)
        else:
            res_score[i] = np.sum(np.abs(ret_FA)) + np.sum(np.abs(ret_MD))
            finish_status[i] = 1

        # calculate threshold to detect miss or false up in segmentation value
        miss_th = int(0.8 * n) + add_to_miss_th

        # find indices of FA (false up) and MD (miss of up)
        if ignore_FA >= max_ind:
            FA = False
        else:
            FA = np.min(ret_FA[ignore_FA:]) < -miss_th
        if ignore_MD >= max_ind:
            MD = False
        else:
            MD = np.max(ret_MD[ignore_MD:]) > miss_th
        i_FA = max_ind
        i_MD = max_ind
        if FA:
            i_FA = np.where(ret_FA[ignore_FA:] < -miss_th)[0][0]
        if MD:
            i_MD = np.where(ret_MD[ignore_MD:] > miss_th)[0][0]

        i_FA = i_FA + ignore_FA
        i_MD = i_MD + ignore_MD

        finish_status[i] = 1

        # cases to deal with different conditions about i_FA and i_MD
        if np.min([i_FA, i_MD]) >= max_ind:
            # if the next FA or MD event are after max_id don't try any more cases and update minimal score if achived
            min_res_score = min(min_res_score, res_score[i])
        elif np.sum(np.abs(ret_FA[:ignore_FA])) + np.sum(np.abs(ret_MD[:ignore_MD])) > min_res_score:
            # do nothing if the calculated sum of FA and MD drifts up to ignor_FA and ignore_MD
            # already above minimal score
            pass
        elif i_FA < i_MD:
            # if the next FA event is before the next MD event try 2 options: ignore the FA event or fix it

            # find the next FA event and save in ignore_FA2
            i_FA_ar = ignore_FA + np.where(ret_FA[ignore_FA:] < -miss_th)[0]

            i_FA_ar = np.clip(i_FA_ar, 0, len(ret_FA) - 2)
            i_FA_end_ar = np.where(ret_FA[i_FA_ar + 1] == -miss_th)[0]
            if len(i_FA_end_ar) > 0:
                ignore_FA2 = i_FA_ar[i_FA_end_ar[0]]
            else:
                ignore_FA2 = max_ind

            # create memory of this case with updates ignore_FA index
            memory = {}
            memory["segs"] = np.copy(segs_temp)
            memory["up_inds_unused"] = np.copy(up_inds_unused_temp)
            memory["down_inds_unused"] = np.copy(down_inds_unused_temp)
            memory["add"] = add_to_miss_th + 1
            memory["ignore_FA"] = ignore_FA2
            memory["ignore_MD"] = ignore_MD

            # check if memory is in memory_ar
            memory_in_ar = False
            for memory1 in memory_ar:
                if same_memory(memory1, memory):
                    memory_in_ar = True
                    break

            # if the new case isn't in memory add to memory_ar and update res_score and finish status of the new memory
            if not memory_in_ar:
                memory_ar.append(memory)
                res_score = np.append(res_score, -1)
                finish_status = np.append(finish_status, 0)

            # fix segs_temp to remove unwanted segment
            if np.size(segs_temp, axis=0) > 0:

                # find false up index and the relevant segment
                ind_false_up = np.argmin(np.abs(segs_temp[:, 0] - i_FA - ret_FA[i_FA]))
                false_up = segs_temp[ind_false_up, 0]
                ind_false_seg = np.where(segs_temp[:, 0] == false_up)[0]
                if len(ind_false_seg) > 1:
                    # if there are few fitting segments with the false up, take only the one with
                    # the largest echo length error
                    seg_sizes = segs_temp[ind_false_seg, 1] - segs_temp[ind_false_seg, 0]

                    n_segs = np.round(seg_sizes / tx_signal_len)
                    segs_err = np.abs((n_segs * tx_signal_len - seg_sizes) / n_segs)

                    ind_false_seg = ind_false_seg[np.argmax(segs_err)]
                else:
                    ind_false_seg = ind_false_seg[0]

                # add up and down indices of the false segment to unused indices arrays
                up_inds_unused_temp = np.append(up_inds_unused_temp, segs_temp[ind_false_seg, 0])
                down_inds_unused_temp = np.append(down_inds_unused_temp, segs_temp[ind_false_seg, 1])

                # delete from segs matrix the false segment
                segs1 = np.delete(segs_temp, ind_false_seg, axis=0)

                # save the updated case to memory dictionary
                memory = {}
                memory["segs"] = np.copy(segs1)
                memory["up_inds_unused"] = np.copy(up_inds_unused_temp)
                memory["down_inds_unused"] = np.copy(down_inds_unused_temp)
                memory["add"] = add_to_miss_th
                memory["ignore_FA"] = ignore_FA
                memory["ignore_MD"] = ignore_MD

                # check if memory in memory_ar
                memory_in_ar = False
                for memory1 in memory_ar:
                    if same_memory(memory1, memory):
                        memory_in_ar = True
                        break

                # if the new case isn't in memory add to memory_ar and update res_score and finish status of the
                # new memory
                if not memory_in_ar:
                    memory_ar.append(memory)
                    res_score = np.append(res_score, -1)
                    finish_status = np.append(finish_status, 0)

        elif i_MD < i_FA:
            # if the next MD event is before the next FA event try 2 options: ignore the MD event or fix it

            # find the next FA event and save in ignore_MD2
            i_MD_ar = ignore_MD + np.where(ret_MD[ignore_MD:] > miss_th)[0]
            i_MD_ar = np.clip(i_MD_ar, 0, len(ret_MD) - 2)
            i_MD_end_ar = np.where(ret_MD[i_MD_ar + 1] == miss_th)[0]
            if len(i_MD_end_ar) > 0:
                ignore_MD2 = i_MD_ar[i_MD_end_ar[0]]
            else:
                ignore_MD2 = max_ind

            # create memory of this case with updates ignore_MD index
            memory = {}
            memory["segs"] = np.copy(segs_temp)
            memory["up_inds_unused"] = np.copy(up_inds_unused_temp)
            memory["down_inds_unused"] = np.copy(down_inds_unused_temp)
            memory["add"] = add_to_miss_th + 1
            memory["ignore_FA"] = ignore_FA
            memory["ignore_MD"] = ignore_MD2

            # check if memory is in memory_ar
            memory_in_ar = False
            for memory1 in memory_ar:
                if same_memory(memory1, memory):
                    memory_in_ar = True
                    break

            # if the new case isn't in memory add to memory_ar and update res_score and finish status of the new memory
            if not memory_in_ar:
                memory_ar.append(memory)
                res_score = np.append(res_score, -1)
                finish_status = np.append(finish_status, 0)

            # fix segs_temp to add the missing segment
            if len(up_inds_unused_temp) > 0 and len(down_inds_unused_temp) > 0:

                # find missing up index from unused indices
                ind_miss_up = np.argmin(np.abs(up_inds_unused_temp - i_MD + ret_MD[i_MD]))
                up_ind = up_inds_unused_temp[ind_miss_up]

                # find the best down index for the mising up index with the smallest echo length error
                # setting the error threshold to x3
                diff_ar, n_diff = get_segs_for_min_err(up_ind, down_inds_unused_temp, 3 * threshold)
                ind_miss_down = np.argmin(diff_ar)
                if n_diff[ind_miss_down] > 0 and diff_ar[ind_miss_down] < 9 * threshold:

                    # find the best down index with differance smaller than 9*threshold
                    down_ind = down_inds_unused_temp[ind_miss_down]

                    # creat new segment
                    seg = [up_ind, down_ind]

                    # remove used indices in the new segment from the arrays of the unused up and down indices
                    up_inds_unused_temp = np.delete(up_inds_unused_temp, ind_miss_up)
                    down_inds_unused_temp = np.delete(down_inds_unused_temp, ind_miss_down)

                    # add the new segment to segs_temp
                    segs1 = np.vstack((segs_temp, seg))

                    # save the updated case to memory dictionary
                    memory = {}
                    memory["segs"] = np.copy(segs1)
                    memory["up_inds_unused"] = np.copy(up_inds_unused_temp)
                    memory["down_inds_unused"] = np.copy(down_inds_unused_temp)
                    memory["add"] = add_to_miss_th
                    memory["ignore_FA"] = ignore_FA
                    memory["ignore_MD"] = ignore_MD

                    # check if memory is in memory_ar
                    memory_in_ar = False
                    for memory1 in memory_ar:
                        if same_memory(memory1, memory):
                            memory_in_ar = True
                            break

                    # if the new case isn't in memory add to memory_ar and update res_score and finish status of the
                    # new memory
                    if not memory_in_ar:
                        memory_ar.append(memory)
                        res_score = np.append(res_score, -1)
                        finish_status = np.append(finish_status, 0)

    # find all the unfinished cases
    unfinished = np.where(res_score == -1)[0]

    # if there are unfinished cases (due to time limit) calculate score and update finish status
    if len(unfinished) > 0:
        for i in unfinished:
            memory = memory_ar[i]
            segs = memory["segs"]

            seg_after_alg = seg_mat2segmentation(segs)

            diff = seg_results - seg_after_alg
            diff_FA = np.zeros_like(diff)
            diff_MD = np.zeros_like(diff)
            diff_FA[diff < 0] = -1
            diff_MD[diff > 0] = 1
            ret_FA = np.cumsum(diff_FA)
            ret_MD = np.cumsum(diff_MD)

            n = int(max(tx_signal_len // 3, 3 * threshold))
            ret_FA[n:] = ret_FA[n:] - ret_FA[:-n]
            ret_MD[n:] = ret_MD[n:] - ret_MD[:-n]

            res_score[i] = np.sum(np.abs(ret_FA)) + np.sum(np.abs(ret_MD))
            finish_status[i] = 1

    # find the case with the minimal score = minimal deviation from network output while keeping the condition to not
    # choose echoes with length different from Tx length by 3*threshold
    i = np.argmin(res_score)

    memory = memory_ar[i]
    segs = memory["segs"]
    up_inds_unused = memory["up_inds_unused"]
    down_inds_unused = memory["down_inds_unused"]

    return segs, up_inds_unused, down_inds_unused


# perform comparison of memory directories
def same_memory(memory, memory_new):
    segs = memory["segs"]
    up_inds_unused = memory["up_inds_unused"]
    down_inds_unused = memory["down_inds_unused"]
    add_to_miss_th = memory["add"]
    ignore_FA = memory["ignore_FA"]
    ignore_MD = memory["ignore_MD"]

    segs_new = memory_new["segs"]
    up_inds_unused_new = memory_new["up_inds_unused"]
    down_inds_unused_new = memory_new["down_inds_unused"]
    add_to_miss_th_new = memory_new["add"]
    ignore_FA_new = memory_new["ignore_FA"]
    ignore_MD_new = memory_new["ignore_MD"]

    same = True
    if not same_len_content(segs, segs_new):
        same = False
    if not same_len_content(up_inds_unused, up_inds_unused_new):
        same = False
    if not same_len_content(down_inds_unused, down_inds_unused_new):
        same = False
    if add_to_miss_th != add_to_miss_th_new:
        same = False
    if ignore_FA != ignore_FA_new:
        same = False
    if ignore_MD != ignore_MD_new:
        same = False

    return same


# check if arrays identical in length and values
def same_len_content(ar1, ar2):
    same = True
    if len(ar1) != len(ar2):
        same = False
    elif not np.all(ar1 == ar2):
        same = False

    return same


# transform segs matrix to segmentation of Rx samples
def seg_mat2segmentation(segs):
    min_range = params.DISTANCE_MIN - params.DIST_EARS / 2
    max_range = params.DISTANCE_MAX + params.DIST_EARS / 2

    segs1 = np.round(segs).astype(int)
    # start with no segmentation (all zeros)
    seg_after_alg = np.zeros(params.DATA_LEN_ROUNDED)

    # for each row increase segmentation value by one from start index to end index
    for row in segs1:
        if row[0] + 1 >= min_range * params.DISTANCE2IND and row[0] - 1 <= max_range * params.DISTANCE2IND:
            seg_after_alg[row[0]:row[1]] = seg_after_alg[row[0]:row[1]] + 1

    return seg_after_alg


# for each down index find the minimal echo length error and number of potential consecutive echos in up to down length
# prioritizing minimal number of echos in up to down length while maintaining echo length error under threshold
def get_segs_for_min_err(up_ind, down_ind_ar, threshold):
    signal_len = params.TX_SIGNAL_LENGTH_IND
    max_ind = params.DATA_LEN_ROUNDED

    # set maximal number of consecutive echos
    max_poss_seq_targets = params.MAX_POSS_SEQ_TARGETS

    # find differance for each down index to up index
    ar_diff = down_ind_ar - up_ind

    # find non logical differance
    ind_non_rel = np.where(np.abs(0.5 * max_ind - ar_diff) > 0.5 * max_ind)[0]  # ar_diff<0 or ar_diif>max_ind

    # for each differance calculate differance for both possible echo length errors (shorter or longer)
    ar_diff1 = np.vstack((np.mod(ar_diff, signal_len), signal_len - np.mod(ar_diff, signal_len)))

    # find option with minimal echo length error (from shorter or longer)
    diff = np.min(ar_diff1, axis=0)
    # find the number of possible echos in up to down length
    n_diff = np.round(ar_diff / signal_len).astype(int)

    # find out of limits number of possible echos
    ind_wrong_n = np.where(np.abs(n_diff - 0.5 * max_poss_seq_targets - 0.1) >
                           0.5 * max_poss_seq_targets)[0]  # n_diff<1 or n_diff>max_poss_seq_targets

    # create array of indices with all fitting down indices in down_ind_ar
    ind_non_rel = np.append(ind_non_rel, ind_wrong_n)
    ind_non_rel = np.unique(ind_non_rel)
    ind_rel = [i for i in range(len(ar_diff)) if not (i in ind_non_rel)]

    if len(ind_rel) > 0:
        # create copy of arrays
        n_diff_copy = np.copy(n_diff)
        diff_copy = np.copy(diff)

        # use only fitting down indices
        n_diff = n_diff[ind_rel]
        diff = diff[ind_rel]

        # find the minimal number of possible echos that there is an echo length error under the threshold
        min_n_diff = np.max([1, np.min(n_diff)]).astype(int) - 1
        max_n_diff = np.min([np.max(n_diff), max_poss_seq_targets]).astype(int)

        while True:
            min_n_diff += 1
            diff_arr = np.append(diff[n_diff == min_n_diff], min_n_diff * threshold + 1)
            if np.min(diff_arr) / min_n_diff <= threshold or min_n_diff > max_n_diff:
                break

        # calculate differance per echo (in case of multiple echos from up to down index)
        diff = diff / n_diff

        # if there is a set of up and down indices with error under threshold set the other difference values with
        # greater differance
        if min_n_diff <= max_n_diff:
            diff[n_diff != min_n_diff] = diff[n_diff != min_n_diff] + 3 * threshold

        # fit updated differences in original arrays
        n_diff = n_diff_copy
        diff_copy[ind_rel] = diff
        diff = diff_copy

    # set differance of all unfitting down indices with value larger than threshold
    if np.size(ind_non_rel) > 0:
        diff[ind_non_rel] = 3 * threshold * np.ones(np.size(ind_non_rel))

    return diff, n_diff


# get range differance from true ranges to estimated ranges
def get_range_diff(gt_ranges, est_ranges):
    gt_ranges = np.copy(gt_ranges)
    est_ranges = np.copy(est_ranges)

    # delete filler values
    gt_ranges = gt_ranges[gt_ranges != 0]
    est_ranges = est_ranges[est_ranges != 0]
    max_range = params.DISTANCE_MAX + params.DIST_EARS / 2

    # fix empty ranges arrays
    if len(np.shape(gt_ranges)) == 0:
        gt_ranges = np.array([gt_ranges])

    if len(np.shape(est_ranges)) == 0:
        est_ranges = np.array([est_ranges])

    # simple solution if gt_ranges is only one value
    if len(gt_ranges) == 1:
        if np.size(est_ranges) == 0:
            range_diff = np.array([])
        elif np.size(est_ranges) == 1:
            range_diff = np.abs(gt_ranges[0] - est_ranges[0])
        else:
            range_diff = np.min(np.abs(gt_ranges[0] - est_ranges))
        return range_diff

    # pad est_ranges with out of limit value to make the lengths equal
    if len(est_ranges) < len(gt_ranges):
        est_ranges = np.pad(est_ranges, (0, len(gt_ranges) - len(est_ranges) + 1), constant_values=(0, max_range * 3))

    used = np.zeros(len(est_ranges))
    range_diff = -np.ones(len(gt_ranges))

    # iterative ranges matching based on minimal error
    count = 0.0
    while np.sum(used) < len(gt_ranges):

        # calculate differance for each pair or estimated and true ranges
        diff_mat = np.abs(gt_ranges[0] - est_ranges)
        for i in range(1, len(gt_ranges)):
            diff_mat = np.vstack((diff_mat, np.abs(gt_ranges[i] - est_ranges)))

        # for each true range find the estimated range with minimal differance
        min_diff_ind = np.vstack((np.min(diff_mat, axis=1), np.argmin(diff_mat, axis=1)))

        # if all pair are unique return the differences
        if len(np.unique(min_diff_ind[1, :])) == len(min_diff_ind[1, :]) and count == 0:
            range_diff = np.squeeze(min_diff_ind[0, :])
            return range_diff

        # get indices of the pair with the lowest error
        ind_gt_min = int(np.argmin(min_diff_ind[0, :]))
        ind_est_min = int(min_diff_ind[1, ind_gt_min])

        # mark the used estimated range as used and add differance to output array
        used[ind_est_min] = 1
        range_diff[ind_gt_min] = np.min(min_diff_ind[0, :])

        # replace used true and estimated ranges as non-logical and very large values
        gt_ranges[ind_gt_min] = -max_range * (10 + count / 10)
        est_ranges[ind_est_min] = max_range * (10 + count / 10)
        count += 1

    return range_diff


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        path_name = "end_segment_model_U_Net_120_64_2em4_2em4_07559999999999998_0998_6em9_11_128_1em2_0_4_1em1_2_Mult_2023_04_25_12_11_04_00000.pth"
        csv_file = "summery_res_2023_04_25_12_11_04.csv"
        date_str = "2023_04_25_12"
        args = ["parallel_model_evaluate.py", params.PARAMS_FOLDER + "/" + date_str + "/" + path_name,
                False, params.PARAMS_FOLDER + "/" + date_str + "/" + csv_file]
    str_ar, _, eval_pathes, _ = main(args)
    for str1, eval_path in zip(str_ar, eval_pathes):
        print(eval_path)
        print(str1)
else:
    pass
