import numpy as np
import torch

from torch.utils.data import DataLoader
import CReflectorsGenerator
import CNumpySonarDataSet
import CNumpySonarDataSaver
import random
import params


# check whether dataset is full of unique labels and that there ar no repetitions
def check_dataset(dataset):
    b_size = 100
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)
    label_array = np.zeros(params.LABEL_LEN_3D)
    i = 0
    for data, label, seg_mask in dataloader:
        label_array = np.vstack([label_array, label])
        i = i + b_size
    unique_labels = np.unique(label_array, axis=0)
    all_unique = np.shape(unique_labels)[0] == i + 1
    print("number of non unique", i + 1 - np.shape(unique_labels)[0])
    return all_unique


def main():
    # setting seeds for random procedures
    seed_val = params.SEED
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # setting file names to be used
    db_name = params.DATASET_FOLDER + "/" + params.BASE_DATASET_NAME_3D
    train_db_name = params.DATASET_FOLDER + "/train_" + params.BASE_DATASET_NAME_3D
    val_db_name = params.DATASET_FOLDER + "/val_" + params.BASE_DATASET_NAME_3D
    test_db_name = params.DATASET_FOLDER + "/test_" + params.BASE_DATASET_NAME_3D
    tx_waveform_file_name = params.DATASET_FOLDER + "/" + params.TX_WAVEFORM_FILE_NAME

    torch.set_num_threads(params.MAX_CPUS)

    train_size = params.TRAIN_RATIO
    val_size = params.VALIDATION_RATIO
    test_size = params.TEST_RATIO

    # config problem parameters
    number_of_range_lists = params.NUM_SAMPLES

    # initializing scene data generator
    data_generator = CReflectorsGenerator.CReflectorsGenerator(params.MAX_TARGETS)

    # constant problem parameters
    max_reflectors = params.MAX_TARGETS
    min_reflectors = params.MIN_TARGETS

    min_range_meter = params.DISTANCE_MIN
    max_range_meter = params.DISTANCE_MAX
    min_theta_rad = params.THETA_MIN
    max_theta_rad = params.THETA_MAX
    min_phi_rad = params.PHI_MIN
    max_phi_rad = params.PHI_MAX
    min_R_meter = params.R_TARGET_MIN
    max_R_meter = params.R_TARGET_MAX

    samples_per_range_list = 1
    total_samples = number_of_range_lists * samples_per_range_list

    # test scene data generation

    test_range = np.array([0.102, 0.20986, 0.32232, 0.9123, 1.2290,
                           1.4045, 1.62246, 1.7035, 1.8237, 2])
    test_theta = np.array([0.7365, 1.0986, 2.62232, 1.69123, 0.72290,
                           1.74045, 0.22246, 1.95035, 0.85037, 1.9622])
    test_phi = np.array([1.7365, 2.0986, 0.62232, 0.69123, 0.72290,
                         1.74045, 0.22246, 0.95035, 0.85037, 1.9622])
    test_R = np.array([0.17365, 0.20986, 0.2232, 0.123, 0.2290,
                       0.14045, 0.2246, 0.25035, 0.15037, 0.22])

    tx, rx, range_list, theta_list, phi_list, _, seg_mask, _ = data_generator.create_reflectors_waveform_3D(
        test_range, test_theta, test_phi, test_R)
    np.save(tx_waveform_file_name, tx)

    # start with clean generator
    data_generator.clean()

    # initialize data saver
    data_saver = CNumpySonarDataSaver.NumpyCSonarDataSaver(
        db_name, total_samples, 2 * params.DATA_LEN_3D, params.LABEL_LEN_3D)

    for i in range(0, number_of_range_lists):
        if i % 100 == 0:
            print("iteration {} out of {}".format(i, number_of_range_lists))
        # sample number of targets
        num_of_targets = np.random.randint(min_reflectors, max_reflectors + 1)
        # set initial scene parameters lists
        range_list = np.random.uniform(min_range_meter, max_range_meter,
                                       size=num_of_targets)
        theta_list = np.random.uniform(min_theta_rad, max_theta_rad, size=num_of_targets)
        phi_list = np.random.uniform(min_phi_rad, max_phi_rad, size=num_of_targets)
        R_list = np.random.uniform(min_R_meter, max_R_meter, size=num_of_targets)

        # generate and add samples
        for j in range(0, samples_per_range_list):
            tx_sig, rx_sig, range_list_for_train, theta_list_for_train, phi_list_for_train, R_list_for_train, mask_seg, valid = data_generator.create_reflectors_waveform_3D(
                range_list, theta_list, phi_list, R_list)

            # in case of repetition of ranges the generated scene stated as invalid and performed another attempt to
            # create unique scene
            while not valid:
                print("data invalid retrying")

                range_list = np.random.uniform(min_range_meter, max_range_meter, num_of_targets)
                theta_list = np.random.uniform(min_theta_rad, max_theta_rad, size=num_of_targets)
                phi_list = np.random.uniform(min_phi_rad, max_phi_rad, size=num_of_targets)
                R_list = np.random.uniform(min_R_meter, max_R_meter, size=num_of_targets)

                tx_sig, rx_sig, range_list_for_train, theta_list_for_train, phi_list_for_train, R_list_for_train, mask_seg, valid = data_generator.create_reflectors_waveform_3D(
                    range_list, theta_list, phi_list, R_list)

            # assert for any inconsistency between generated scene parameters and limits
            assert (len(range_list_for_train) == max_reflectors)
            assert (len(theta_list_for_train) == max_reflectors)
            assert (len(phi_list_for_train) == max_reflectors)
            assert (np.min(range_list_for_train) > 0)

            # set all scene parameters (range, theta, phi) label
            labels = np.hstack((range_list_for_train, theta_list_for_train,
                                phi_list_for_train))

            # save scene sample
            data_saver.add(np.reshape(rx_sig, (-1,)), labels,
                           np.reshape(mask_seg, (-1,)))

    # check the created dataset
    dataset = CNumpySonarDataSet.CNumpySonarDataSet(data_base_file_name=db_name,
                                                    data_sample_len=2 * params.DATA_LEN_3D,
                                                    label_len=params.LABEL_LEN_3D,
                                                    params=params, transform=None,
                                                    target_transform=None)

    data_size = dataset.__len__()
    print("db sample size", data_size)
    assert total_samples == data_size
    train_val_set, test_set = torch.utils.data.random_split(
        dataset, [int(data_size * (train_size + val_size)), int(data_size * test_size)])
    print("train db sample size ", train_val_set.__len__())
    print("test db sample size ", test_set.__len__())
    data, label, seg_mask = dataset[99]
    train_dataloader = DataLoader(train_val_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

    train_features, train_labels, seg_mask = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    all_unique = check_dataset(dataset)
    if all_unique:
        print("data set valid")
    else:
        print("data set invalid")

    # split created dataset to train, validation and test sets and save them separately
    train_set, val_set = torch.utils.data.random_split(
        train_val_set, [int(data_size * train_size), int(data_size * val_size)])

    train_data_saver = CNumpySonarDataSaver.NumpyCSonarDataSaver(
        train_db_name, int(data_size * train_size), 2 * params.DATA_LEN_3D,
        params.LABEL_LEN_3D)
    val_data_saver = CNumpySonarDataSaver.NumpyCSonarDataSaver(
        val_db_name, int(data_size * val_size), 2 * params.DATA_LEN_3D,
        params.LABEL_LEN_3D)
    test_data_saver = CNumpySonarDataSaver.NumpyCSonarDataSaver(
        test_db_name, int(data_size * test_size), 2 * params.DATA_LEN_3D,
        params.LABEL_LEN_3D)

    for i in range(len(train_set)):
        data, label, seg_mask = train_set[i]
        train_data_saver.add(data, label, seg_mask)

    for i in range(len(val_set)):
        data, label, seg_mask = val_set[i]
        val_data_saver.add(data, label, seg_mask)

    for i in range(len(test_set)):
        data, label, seg_mask = test_set[i]
        test_data_saver.add(data, label, seg_mask)

    # check the created datasets
    train_dataset = CNumpySonarDataSet.CNumpySonarDataSet(train_db_name,
                                                          2 * params.DATA_LEN_3D,
                                                          params.LABEL_LEN_3D, params,
                                                          transform=None)

    data_size = train_dataset.__len__()
    print("db sample size", data_size)
    data, label, seg_mask = train_dataset[99]
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_features, train_labels, seg_mask = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")


if __name__ == '__main__':
    main()
    print("done")
