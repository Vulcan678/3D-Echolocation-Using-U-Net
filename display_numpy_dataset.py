import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import CNumpySonarDataSet
import NormalSignal
import params

# display dataset samples
def main():

    # load dataset
    db_name = params.DATASET_FOLDER+"/"+params.BASE_DATASET_NAME_3D
    dataset = CNumpySonarDataSet.CNumpySonarDataSet(db_name, 2 * params.DATA_LEN_3D, params.LABEL_LEN_3D, params,
                                                    transform=None)

    data_size = dataset.__len__()
    print("db sample size", data_size)

    # create DataLoader to iterate over samples
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for waveform, labels, seg_mask in dataloader:
        print("range ", labels)

        waveforms = np.vstack((waveform[0][:params.DATA_LEN_3D], waveform[0][params.DATA_LEN_3D:]))
        seg_masks = np.vstack((seg_mask[0][:params.DATA_LEN_3D], seg_mask[0][params.DATA_LEN_3D:]))

        # display for each ear separately
        for waveform1, seg_mask1 in zip(waveforms, seg_masks):
            plt.figure()

            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            ax1.plot(waveform1, )
            ax2.plot(seg_mask1, 'k-')

            ax1.set_ylabel('Signal')
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=1e-1)
            ax2.set_ylabel('Segmentation', color='k')

            plt.show()

            # display after logarithmic transformation for scaling
            coxbox_trans = NormalSignal.BoxCox(params.RX_SIGNAL_REF_GAIN,
                                               lamb=params.BOX_COX_LAMBDA)

            norm_wave = coxbox_trans(waveform1)

            plt.figure()

            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            ax1.plot(norm_wave, )
            ax2.plot(seg_mask1, 'k-')

            ax1.set_ylabel('Signal')
            ax2.set_ylabel('Segmentation', color='k')

            plt.show()


if __name__ == '__main__':
    main()
    print("done")
