import matplotlib.pyplot as plt
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import NormalSignal

import time
from datetime import datetime
import model_metrcis
from model import Unet1d_all
from utils import *
import seg_loss_func
import parallel_model_evaluate
import re
import sys
import pickle
import pandas as pd
from scipy.optimize import curve_fit
import glob
import os

import params

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce
from torch.distributed import ReduceOp

from hyperopt import tpe, hp, fmin, Trials, STATUS_OK


# setup an DDP for multi-GPU run
def ddp_setup(rank, world_size):
    port = '26786'  # random unused number

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


# creating a class for training
class Trainer:
    # initialaizing training parameters
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.StepLR,
            loss_func: seg_loss_func,
            gpu_id: int,
            save_every: int,
            world_size: int,
            batch_size: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.save_every = save_every
        self.world_size = world_size
        self.batch_size = batch_size
        self.distributed = self.world_size > 1

        # for multi-GPU run setup a DDP
        if not self.distributed:
            self.model = model
        else:
            self.model = DDP(model, device_ids=[gpu_id])

    # creating a base file name according to network type and training characteristics
    def _param_file_name(self, max_epochs, learning_rate, fWeight_decay, betas,
                         lr_scheduler_min_lr, norm_ref, norm_lambda, kernel_size, max_ch,
                         n_head, drop_prob, n_trans_layers, att_type, model_name,
                         date_str):

        file_name = "segment_model"

        max_epochs_name = str(max_epochs)
        if learning_rate > 0:
            lr_power = int(-np.floor(np.log10(learning_rate)))
            lr_coef = int(learning_rate * (10 ** lr_power))
            lr_name = str(lr_coef) + "em" + str(lr_power)
        else:
            lr_name = "0"
        if fWeight_decay > 0:
            wd_power = int(-np.floor(np.log10(fWeight_decay)))
            wd_coef = int(fWeight_decay * (10 ** wd_power))
            wd_name = str(wd_coef) + "em" + str(wd_power)
        else:
            wd_name = "0"
        beta1_name = re.sub("[.]+", "", str(np.round(betas[0], 4)))
        beta2_name = re.sub("[.]+", "", str(np.round(betas[1], 4)))
        if lr_scheduler_min_lr > 0:
            min_lr_power = int(-np.floor(np.log10(lr_scheduler_min_lr)))
            min_lr_coef = int(lr_scheduler_min_lr * (10 ** min_lr_power))
            min_lr_name = str(min_lr_coef) + "em" + str(min_lr_power)
        else:
            min_lr_name = "0"
        if norm_ref == 1:
            norm_ref_name = "1"
        else:
            nr_power = int(np.floor(np.log10(norm_ref)))
            nr_coef = int(norm_ref * (10 ** (-nr_power)))
            if norm_ref > 1:
                norm_ref_name = str(nr_coef) + "e" + str(nr_power)
            else:
                norm_ref_name = str(nr_coef) + "em" + str(int(np.abs(nr_power)))
        if norm_lambda == 0:
            norm_lambda_name = "0"
        else:
            nl_power = int(np.floor(np.log10(norm_lambda)))
            nl_coef = int(norm_lambda * (10 ** (-nl_power)))
            if norm_lambda > 1:
                norm_lambda_name = str(nl_coef) + "e" + str(nl_power)
            else:
                norm_lambda_name = str(nl_coef) + "em" + str(int(np.abs(nl_power)))

        if drop_prob > 0:
            drop_power = int(-np.floor(np.log10(drop_prob)))
            drop_coef = int(drop_prob * (10 ** drop_power))
            drop_name = str(drop_coef) + "em" + str(drop_power)
        else:
            drop_name = "0"

        path_name = file_name + "_" + model_name + "_" + max_epochs_name
        path_name = path_name + "_" + str(self.batch_size) + "_" + lr_name + "_" + wd_name
        path_name = path_name + "_" + beta1_name + "_" + beta2_name + "_" + min_lr_name
        path_name = path_name + "_" + str(kernel_size) + "_" + str(max_ch)
        path_name = path_name + "_" + norm_ref_name + "_" + norm_lambda_name
        path_name = path_name + "_" + str(n_head) + "_" + drop_name
        path_name = path_name + "_" + str(n_trans_layers) + "_" + att_type
        path_name = path_name + "_" + date_str + ".pth"

        return path_name

    # run a batch training and update optimizer and loss
    def _run_batch(self, data, seg_mask):
        self.optimizer.zero_grad()
        net_input = data.unsqueeze(1)
        output = self.model(net_input)
        seg_mask = seg_mask.long()
        loss = self.loss_func(output, seg_mask)
        loss.backward()
        self.optimizer.step()

        return loss

    # run an epoch of distributed training between GPU's based of batches
    def _run_epoch(self, epoch):
        # print training process
        print("[GPU {self.gpu_id}] Epoch {epoch} | Batch size: {self.batch_size} | Steps: {len(self.train_data)}")
        if self.gpu_id == 0:
            self.train_data.sampler.set_epoch(epoch)

        # enter training mode
        self.model.train()
        count = 0
        loss_sum = 0

        # iterate over batches, for each ear
        for data, labels, seg_mask in self.train_data:
            data = data.to(self.gpu_id)
            seg_mask = seg_mask.to(self.gpu_id)
            loss_sum += self._run_batch(data[:, :params.DATA_LEN_ROUNDED_3D], seg_mask[:, :params.DATA_LEN_ROUNDED_3D])
            count += 1
            loss_sum += self._run_batch(data[:, params.DATA_LEN_ROUNDED_3D:], seg_mask[:, params.DATA_LEN_ROUNDED_3D:])
            count += 1

        # update learning rate scheduler
        self.scheduler.step()
        return loss_sum / count

    # run a batch validation, calculate loss and dice loss
    def _run_val_batch(self, data, seg_mask):
        net_input = data.unsqueeze(1)
        output = self.model(net_input)
        seg_mask = seg_mask.long()
        loss = self.loss_func(output, seg_mask)

        seg_res = output.squeeze(1)
        seg_res = torch.argmax(seg_res, dim=1)
        dice = model_metrcis.dice_coef_multilabel(seg_mask.flatten(), seg_res.flatten())

        return loss, dice

    # run an epoch of distributed validation between GPU's based of batches
    def _run_val_epoch(self, epoch):
        if self.gpu_id == 0:
            self.val_data.sampler.set_epoch(epoch)

        # enter evaluation mode
        self.model.eval()
        count = 0
        loss_sum = torch.Tensor([0]).to(self.gpu_id)
        dice_coef = torch.Tensor([0]).to(self.gpu_id)

        # don't consider the steps in gradient calculation, evaluate dice and loss
        with torch.no_grad():
            for data, labels, seg_mask in self.train_data:
                data = data.to(self.gpu_id)
                seg_mask = seg_mask.to(self.gpu_id)
                loss, dice = self._run_val_batch(data[:, :params.DATA_LEN_ROUNDED_3D],
                                                 seg_mask[:, :params.DATA_LEN_ROUNDED_3D])
                loss_sum += loss.to(self.gpu_id)
                dice_coef += dice.to(self.gpu_id)
                count += 1
                loss, dice = self._run_val_batch(data[:, params.DATA_LEN_ROUNDED_3D:],
                                                 seg_mask[:, params.DATA_LEN_ROUNDED_3D:])
                loss_sum += loss.to(self.gpu_id)
                dice_coef += dice.to(self.gpu_id)
                count += 1
        return loss_sum / count, dice_coef / count

    # perform the train procedure
    def train(self, max_epochs: int, learning_rate: float, fWeight_decay: float,
              betas: tuple, lr_scheduler_min_lr: float, norm_ref: float,
              norm_lambda: float, kernel_size: int, max_ch: int, n_head: int,
              drop_prob: float, n_trans_layers: int, att_type: str, model_name: str,
              date_str: str):

        # create path name for created files
        path_name = self._param_file_name(max_epochs, learning_rate, fWeight_decay, betas,
                                          lr_scheduler_min_lr, norm_ref, norm_lambda,
                                          kernel_size, max_ch, n_head, drop_prob,
                                          n_trans_layers, att_type, model_name, date_str)

        # initialize loss and dice arrays
        loss_ar = np.zeros(max_epochs)
        loss_val_ar = np.zeros(max_epochs)
        dice_score_ar = np.zeros(max_epochs)

        # start timing of the whole training phase
        if self.gpu_id == 0 or not self.distributed:
            start_train_time = time.time()

        # perform training by epochs, until reaching max_epochs
        for epoch in range(max_epochs):
            # start timing of the epoch
            start_time = time.time()

            # perform training and validation for this epoch
            loss = self._run_epoch(epoch)
            val_loss, dice = self._run_val_epoch(epoch)

            # sum loss and dice values from distributed GPU evaluations of the epoch
            if self.distributed:
                all_reduce(loss, ReduceOp.SUM)
                all_reduce(val_loss, ReduceOp.SUM)
                all_reduce(dice, ReduceOp.SUM)

            # calculate mean loss and dice value between distributed GPU evaluations of the epoch
            loss = loss.detach().cpu().numpy() / max(self.world_size, 1)
            val_loss = val_loss.detach().cpu().numpy()[0] / max(self.world_size, 1)
            dice = dice.detach().cpu().numpy()[0] / max(self.world_size, 1)

            # add loss and dice values for analysis
            loss_ar[epoch] = loss
            loss_val_ar[epoch] = val_loss
            dice_score_ar[epoch] = dice

            # print epoch statistics and save progress
            if self.gpu_id == 0 or not self.distributed:
                run_time = time.time() - start_time

                print("Epoch: {0:d}, Loss: {1:.4e}, loss val: {2:.4e}, Time[sec]: {3:.2e}".format(
                    epoch, loss, val_loss, run_time))
                print("dice score {0:.4e}".format(dice))

                if epoch % self.save_every == 0 and epoch > 0:
                    path = params.PARAMS_FOLDER + "/" + date_str[:-5] + "/ep_" + \
                           str(epoch).zfill(3) + "_" + path_name

                    save_model(self.model, path, dist=False)

            # evaluate loss and dice propagation according to assumed convergence curve
            if epoch % 10 == 0 and epoch >= 50:
                prev = epoch
                x = np.arange(epoch - prev, epoch + 1)
                popt, loss_sd = curve_fit_sd(x, loss_ar[epoch - prev:epoch + 1],
                                             f_loss_model, (-0.04, 0.01, 0.01))
                popt, loss_val_sd = curve_fit_sd(x, loss_val_ar[epoch - prev:epoch + 1],
                                                 f_loss_model, (-0.04, 0.01, 0.01))
                popt, dice_sd = curve_fit_sd(x, 1 - dice_score_ar[epoch - prev:epoch + 1],
                                             f_loss_model, (-0.04, 0.01, 0.01))

                prev = 10

                loss_ar_rel = np.log10(loss_ar[epoch - prev:epoch + 1])
                dice_score_ar_rel = np.log10(1 - dice_score_ar[epoch - prev:epoch + 1])

                loss_var = np.var(loss_ar_rel)
                dice_var = np.var(dice_score_ar_rel)

                loss_fluctuations = np.max(np.diff(loss_ar_rel))
                dice_fluctuations = np.max(np.diff(dice_score_ar_rel))

                # stop training progress due to crazy (bad) convergence
                if loss_val_sd > 0.2 and dice_sd > 0.2:
                    if self.gpu_id == 0 or not self.distributed:
                        print("break run due to unstability")
                        print("loss SD: " + str(loss_sd) + "\n" +
                              "loss_val SD: " + str(loss_val_sd) + "\n" +
                              "dice SD: " + str(dice_sd) + "\n" +
                              "log10(loss) var: " + str(loss_var) + "\n" +
                              "log10(1-dice) var: " + str(dice_var) + "\n" +
                              "log10(loss) fluc: " + str(loss_fluctuations) + "\n" +
                              "log10(1-dice) fluc: " + str(dice_fluctuations))

                    break

        # at the end of training save model, print and save evaluation progress and run time
        if self.gpu_id == 0 or not self.distributed:
            path = params.PARAMS_FOLDER + "/" + date_str[:-5] + "/end_" + path_name
            save_model(self.model, path, dist=False)  # self.distributed)

            print("Loss array")
            print(loss_ar[loss_ar > 0])
            print("Loss val array")
            print(loss_val_ar[loss_val_ar > 0])
            print("Dice Score array")
            print(dice_score_ar[loss_val_ar > 0])

            run_time = time.time() - start_train_time

            print(run_time)

            data = (loss_ar, loss_val_ar, dice_score_ar, run_time)
            with open(params.PARAMS_FOLDER + "/" + date_str[:-5] + "/res_" + date_str + '.pickle', 'wb') as f:
                pickle.dump(data, f)

            plt.rcParams["figure.figsize"] = params.FIG_SIZE

            plt.figure()
            plt.plot(loss_ar, label='Training Loss')
            plt.plot(loss_val_ar, color='red',
                     linestyle='dotted', label='Validation Loss')
            plt.xlabel("Num Epoch", fontsize=26)
            plt.xticks(fontsize=20)
            plt.ylabel("Loss Value", fontsize=26)
            plt.yticks(fontsize=20)
            plt.yscale("log")
            plt.grid(True)
            plt.legend(loc='upper right', fontsize=26)
            plt.savefig(params.PARAMS_FOLDER + "/" +
                        date_str[:-5] + '/loss_' + path_name[:-4] + '.png')
            plt.show()

            plt.figure()
            plt.plot(dice_score_ar)
            plt.xlabel("Num Epoch", fontsize=26)
            plt.xticks(fontsize=20)
            plt.ylabel("Validation Dice Score", fontsize=26)
            plt.yticks(fontsize=20)
            plt.grid(True)
            plt.savefig(params.PARAMS_FOLDER + "/" +
                        date_str[:-5] + '/dice_' + path_name[:-4] + '.png')
            plt.show()


# load training objects: parameters, datasets, model, training objects
def load_train_objs(norm_ref: float, norm_lambda: float, kernel_size: int, max_ch: int,
                    learning_rate: float, fWeight_decay: float, betas: tuple,
                    lr_max_steps: int, lr_scheduler_min_lr: float, n_head: int,
                    drop_prob: float, n_trans_layers: int, att_type: str, model_name: str):
    # import parameters
    dataset_folder = params.DATASET_FOLDER
    dataset_name = params.BASE_DATASET_NAME_3D
    data_sample_len = 2 * params.DATA_LEN_3D
    label_len = params.LABEL_LEN_3D
    number_of_classes = params.MAX_TARGETS + 1  # 1 for no target

    # load your dataset
    norm_trans = NormalSignal.BoxCox(norm_ref, lamb=norm_lambda)
    train_set, val_set, test_set = create_dataset(dataset_folder, dataset_name,
                                                           data_sample_len, label_len,
                                                           params, transform=norm_trans)

    # load your model
    if model_name == "U_Net":
        model = Unet1d_all.U_Net(signal_ch=1, output_ch=number_of_classes,
                                 kernel_size=kernel_size, ch_max=max_ch)
    elif model_name == "AttU_Net":
        model = Unet1d_all.AttU_Net(signal_ch=1, output_ch=number_of_classes,
                                    kernel_size=kernel_size, ch_max=max_ch)
    elif model_name == "TransU_Net":
        model = Unet1d_all.TransU_Net(signal_ch=1, output_ch=number_of_classes,
                                      kernel_size=kernel_size, ch_max=max_ch,
                                      n_head=n_head, drop_prob=drop_prob,
                                      n_layers=n_trans_layers, att_type=att_type)
    elif model_name == "ClassicTransU_Net":
        model = Unet1d_all.ClassicTransU_Net(signal_ch=1, output_ch=number_of_classes,
                                             kernel_size=kernel_size, ch_max=max_ch,
                                             n_head=n_head, drop_prob=drop_prob,
                                             n_layers=n_trans_layers)
    elif model_name == "R2AttU_Net":
        model = Unet1d_all.R2AttU_Net(signal_ch=1, output_ch=number_of_classes,
                                      kernel_size=kernel_size, ch_max=max_ch)
    elif model_name == "R2TransU_Net":
        model = Unet1d_all.R2TransU_Net(signal_ch=1, output_ch=number_of_classes,
                                        kernel_size=kernel_size, ch_max=max_ch,
                                        n_head=n_head, drop_prob=drop_prob,
                                        n_layers=n_trans_layers, att_type=att_type)

    # evaluate number of training parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of training parameters", pytorch_total_params)

    # initializing optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=fWeight_decay, betas=betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=lr_max_steps,
                                                           eta_min=lr_scheduler_min_lr)

    return train_set, val_set, model, optimizer, scheduler


# initialize dataloader for distributed or undistributed evaluations
def prepare_dataloader(dataset: Dataset, batch_size: int, world_size: int):
    if world_size <= 1:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset)
        )
    return dataloader


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int,
         norm_ref: float, norm_lambda: float, kernel_size: int, max_ch: int,
         learning_rate: float, fWeight_decay: float, betas: tuple, lr_max_steps: int,
         lr_scheduler_min_lr: float, n_head: int, drop_prob: float, n_trans_layers: int,
         att_type: str, loss_func: object, model_name: str, date_str: str):
    # setup DDP for distributed evaluations and set limit on CPU core use
    if world_size > 1:
        ddp_setup(rank, world_size)
    torch.set_num_threads(int(min(params.MAX_CPUS / max(world_size, 1), 2)))

    # load training objects
    train_set, val_set, model, optimizer, scheduler = load_train_objs(norm_ref, norm_lambda,
                                                                      kernel_size, max_ch,
                                                                      learning_rate,
                                                                      fWeight_decay, betas,
                                                                      lr_max_steps,
                                                                      lr_scheduler_min_lr,
                                                                      n_head, drop_prob,
                                                                      n_trans_layers,
                                                                      att_type, model_name)

    # initialize dataloaders
    train_data = prepare_dataloader(train_set, batch_size, world_size)
    val_data = prepare_dataloader(val_set, batch_size, world_size)

    # initialize trainer
    trainer = Trainer(model, train_data, val_data, optimizer, scheduler,
                      loss_func, rank, save_every, world_size, batch_size)

    # perform training
    trainer.train(total_epochs, learning_rate, fWeight_decay, betas, lr_scheduler_min_lr,
                  norm_ref, norm_lambda, kernel_size, max_ch, n_head, drop_prob,
                  n_trans_layers, att_type, model_name, date_str)

    # at the end close all processes
    if world_size > 1:
        destroy_process_group()


# loss convergence model
def f_loss_model(x, a, b, c):
    return np.exp(a * x + b) + c


# perform curve fitting for convergence evaluation
def curve_fit_sd(x, y, f, p0):
    try:
        popt, pcov = curve_fit(
            f=f,  # model function
            xdata=x,  # x data
            ydata=y,  # y data
            p0=p0,  # initial value of the parameter
            maxfev=10000
        )
        y_est = f(x, popt[0], popt[1], popt[2])
        sd = np.sqrt(np.mean((y - y_est) ** 2))
    except:
        # if convergence failed return initial guess with large SD
        popt = p0
        sd = 10

    return popt, sd


# perform training and after wards analyze results of the training setup and add to CSV summary file
def train_objective(train_params):
    # extract passed training parameters
    batch_size = int(train_params['batch_size'])
    learning_rate = train_params['learning_rate']
    fWeight_decay = train_params['fWeight_decay']
    beta1 = 1 - train_params['1-beta1']
    beta2 = 1 - train_params['1-beta2']
    lr_scheduler_min_lr = min(learning_rate, train_params['lr_scheduler_min_lr'])
    loss_func = train_params['loss_func']

    num_epochs = int(train_params['num_epochs'])
    kernel_size = int(train_params['kernel_size'])
    max_ch = int(train_params['max_ch'])
    norm_ref = train_params['norm_ref']
    norm_lambda = train_params['norm_lambda']
    att_type = train_params['att_type']
    n_head = int(train_params['n_head'])
    drop_prob = train_params['drop_prob']
    n_trans_layers = int(train_params['n_trans_layers'])

    date_str = train_params['date_str']
    world_size = train_params['world_size']
    save_every = train_params['save_every']
    model_name = train_params['model_name']

    betas = (beta1, beta2)

    # create or update CSV file with training setup parameters and training results
    columns_names = ['run_num', 'model name', 'num epochs', 'epoch stop', 'batch size',
                     'learning rate', 'weight_decay', 'beta1', 'beta2', 'min lr',
                     'kernel size', 'max ch', 'norm ref', 'norm lambda', 'n heads',
                     'drop prob', 'n trans layers', 'attention type', 'loss_func', 'loss',
                     'loss_val', 'dice', 'loss SD', 'loss_val SD', 'dice SD',
                     'Avg Error', 'Avg FA rate', 'Avg MD rate', 'run time',
                     'eval run time']

    try:
        df = pd.read_csv(params.PARAMS_FOLDER + "/" + date_str + "/summery_res_" + date_str + ".csv")
        count = int(np.array(df['run_num'])[-1])
    except:
        df = pd.DataFrame(columns=columns_names)
        count = -1

    count += 1
    print(count)

    # create unique date for specific training setup
    date_str1 = date_str + "_" + str(count).zfill(4)

    if world_size <= 1:
        # perform training on single GPU or on CPU
        main(params.DEVICE, world_size, save_every, num_epochs, batch_size,
             norm_ref, norm_lambda, kernel_size, max_ch, learning_rate,
             fWeight_decay, betas, lr_max_steps, lr_scheduler_min_lr, n_head,
             drop_prob, n_trans_layers, att_type, loss_func, model_name, date_str1)
    else:
        # perform distributed training on multiple GPU's
        mp.spawn(main, args=(world_size, save_every, num_epochs, batch_size,
                             norm_ref, norm_lambda, kernel_size, max_ch,
                             learning_rate, fWeight_decay, betas, lr_max_steps,
                             lr_scheduler_min_lr, n_head, drop_prob,
                             n_trans_layers, att_type, loss_func, model_name,
                             date_str1),
                 nprocs=world_size)

    # load last achieved model and analyze convergence process
    with open(params.PARAMS_FOLDER + "/" + date_str + "/res_" + date_str1 + '.pickle', 'rb') as f:
        data = pickle.load(f)
        loss = data[0][data[0] > 0][-1]
        loss_val = data[1][data[1] > 0][-1]
        dice = data[2][data[2] > 0][-1]

        x = np.arange(1, len(data[0]) + 1)

        popt, loss_sd = curve_fit_sd(x, data[0], f_loss_model, (-0.04, 0.01, 0.01))
        popt, loss_val_sd = curve_fit_sd(x, data[1], f_loss_model, (-0.04, 0.01, 0.01))
        popt, dice_sd = curve_fit_sd(x, 1 - data[2], f_loss_model, (-0.04, 0.01, 0.01))

    # perform evalution of the model performance
    avg_err, avg_false_target_rate, miss_detection_rate = 1, 1, 1
    eval_run_time = -1

    # perform evaluation only on results with good metrics
    if len(data[0][data[0] > 0]) > 0.7 * num_epochs and dice >= 0.95:
        list_of_files = glob.glob(params.PARAMS_FOLDER + "/" + date_str + '/end_*_' + date_str1 + '.pth')
        path_name = list_of_files[-1]

        args = ["", path_name, False]
        _, evaluate_params, _, eval_run_time = parallel_model_evaluate.main(args)
        avg_err, avg_false_target_rate, miss_detection_rate = evaluate_params[0]

    # add training setup to CSV and the training's convergence process and model's performance
    list1 = [model_name, num_epochs, len(data[0][data[0] > 0]), batch_size,
             learning_rate, fWeight_decay, beta1, beta2, lr_scheduler_min_lr,
             kernel_size, max_ch, norm_ref, norm_lambda, n_head, drop_prob,
             n_trans_layers, att_type, loss_func, loss, loss_val, dice, loss_sd,
             loss_val_sd, dice_sd, avg_err, avg_false_target_rate, miss_detection_rate,
             data[3], eval_run_time]

    columns_names = ['run_num', 'model name', 'num epochs', 'epoch stop', 'batch size',
                     'learning rate', 'weight_decay', 'beta1', 'beta2', 'min lr',
                     'kernel size', 'max ch', 'norm ref', 'norm lambda', 'n heads',
                     'drop prob', 'n trans layers', 'attention type', 'loss_func', 'loss',
                     'loss_val', 'dice', 'loss SD', 'loss_val SD', 'dice SD',
                     'Avg Error', 'Avg FA rate', 'Avg MD rate', 'run time',
                     'eval run time']

    print(list1)

    list1 = [count] + list1
    dict1 = {columns_names[i]: [list1[i]] for i in range(len(columns_names))}

    df = pd.concat((df, pd.DataFrame(dict1)), axis=0, ignore_index=True)
    df.to_csv(params.PARAMS_FOLDER + "/" + date_str +
              "/summery_res_" + date_str + ".csv", index=False)

    return {
        'loss': np.log10(1 - dice),
        'dice': dice,
        'avg_err': avg_err,
        'avg_false_target_rate': avg_false_target_rate,
        'miss_detection_rate': miss_detection_rate,
        'status': STATUS_OK}


if __name__ == "__main__":
    # set seed
    seed_val = params.SEED
    seed_everything(seed=seed_val)

    # allow run from saved default parameters or user defined parameters
    args = sys.argv
    if len(args) < 19:
        learning_rate = params.LEARNING_RATE
        fWeight_decay = params.FWEIGHT_DECAY
        betas = params.BETAS
        lr_scheduler_min_lr = params.LR_SCHEDUALER_MIN_LR
        num_epochs = params.NUM_EPOCHS
        kernel_size = params.KERNEL_SIZE
        max_ch = params.MAX_CH
        norm_ref = params.RX_SIGNAL_REF_GAIN
        norm_lambda = params.BOX_COX_LAMBDA
        save_every = params.SAVE_EVERY
        batch_size = params.BATCH_SIZE
        n_head = params.N_TRANS_HEAD
        drop_prob = params.DROP_PROB
        n_trans_layers = params.N_TRANS_LAYERS
        att_type = params.ATT_TYPE
        model_name = params.MODEL_NAME
    else:
        learning_rate = float(args[1])
        fWeight_decay = float(args[2])
        betas = (float(args[3]), float(args[4]))
        lr_scheduler_min_lr = float(args[6])
        num_epochs = int(args[7])
        kernel_size = int(args[8])
        max_ch = int(args[9])
        norm_ref = float(args[10])
        norm_lambda = float(args[11])
        save_every = int(args[12])
        batch_size = int(args[13])
        n_head = int(args[14])
        drop_prob = float(args[15])
        n_trans_layers = int(args[16])
        att_type = args[17]
        model_name = args[18]

    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    lr_max_steps = num_epochs
    world_size = torch.cuda.device_count()

    # one run
    """
    if world_size <= 1:
        main(params.DEVICE, world_size, save_every, num_epochs, batch_size, norm_ref,
             norm_lambda, kernel_size, max_ch, learning_rate, fWeight_decay, betas,
             lr_max_steps, lr_scheduler_min_lr, n_head, drop_prob,
             n_trans_layers, att_type, model_name, date_str)
    else:
        mp.spawn(main, args=(world_size, save_every, num_epochs, batch_size, norm_ref,
                             norm_lambda, kernel_size, max_ch, learning_rate,
                             fWeight_decay, betas, lr_max_steps, lr_scheduler_min_lr, 
                             n_head, drop_prob, n_trans_layers, att_type, model_name, 
                             date_str), nprocs=world_size)
    """

    # multiple runs

    path = os.path.join(params.PARAMS_FOLDER, date_str)
    os.mkdir(path)

    # create DB of all possible training setups
    bce_and_dice_loss = seg_loss_func.DiceBCELoss()
    dice_loss = seg_loss_func.DiceLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    iou_loss = seg_loss_func.IoULoss()
    focal_loss = seg_loss_func.FocalLoss()
    TverskyLoss = seg_loss_func.TverskyLoss()

    batch_size_ar = [64]
    lr_ar = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    wd_ar = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-9]
    beta1_ar = [0.5, 0.7, 0.9, 0.99]
    beta2_ar = [0.9, 0.999, 0.9999]
    min_lr_ar = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    loss_funcs = [bce_and_dice_loss, dice_loss, cross_entropy_loss, iou_loss, focal_loss, TverskyLoss]

    model_ar = ["TransU_Net"]  # ["U_Net", "AttU_Net", "TransU_Net", "ClassicTransU_Net", "R2AttU_Net", "R2TransU_Net"]
    epochs_ar = [num_epochs]  # 120
    kernal_ar = [kernel_size]  # 11
    max_ch_ar = [max_ch]  # 128 (>=128)
    ref_ar = [norm_ref]  # 0.01
    lambda_ar = [norm_lambda]  # 0

    # transformer params
    att_type_ar = ["Add"]  # ["Add", "Mult"]
    n_head_ar = [n_head]  # 4 [2, 4, 6, 8, ...]
    drop_prob_ar = [drop_prob]  # 0.2
    n_trans_layers_ar = [n_trans_layers]  # 2

    count = -1
    # run for different model types
    for model_name in model_ar:
        count += 1
        if model_name.find("Trans") == -1:
            att_type_ar1 = [params.ATT_TYPE]
            n_head_ar1 = [params.N_TRANS_HEAD]
            drop_prob_ar1 = [params.DROP_PROB]
            n_trans_layers_ar1 = [params.N_TRANS_LAYERS]
        else:
            att_type_ar1 = att_type_ar
            n_head_ar1 = n_head_ar
            drop_prob_ar1 = drop_prob_ar
            n_trans_layers_ar1 = n_trans_layers_ar

        date_str1 = date_str + "_" + str(count).zfill(4)

        # creating search space for training hyper-parameters optimization
        space = {}
        if len(batch_size_ar) == 1:
            space['batch_size'] = hp.choice('batch_size', batch_size_ar)
        else:
            space['batch_size'] = hp.quniform('batch_size', np.min(batch_size_ar),
                                              np.max(batch_size_ar),
                                              np.min(batch_size_ar))
        if len(lr_ar) == 1:
            space['learning_rate'] = hp.choice('learning_rate', lr_ar)
        else:
            space['learning_rate'] = hp.loguniform('learning_rate',
                                                   np.log(np.min(lr_ar)),
                                                   np.log(np.max(lr_ar)))
        if len(wd_ar) == 1:
            space['fWeight_decay'] = hp.choice('fWeight_decay', wd_ar)
        else:
            space['fWeight_decay'] = hp.loguniform('fWeight_decay',
                                                   np.log(np.min(wd_ar)),
                                                   np.log(np.max(wd_ar)))
        if len(beta1_ar) == 1:
            space['1-beta1'] = hp.choice('1-beta1', 1 - np.array(beta1_ar))
        else:
            space['1-beta1'] = hp.loguniform('1-beta1', np.log(1 - np.max(beta1_ar)),
                                             np.log(1 - np.min(beta1_ar)))
        if len(beta2_ar) == 1:
            space['1-beta2'] = hp.choice('1-beta2', 1 - np.array(beta2_ar))
        else:
            space['1-beta2'] = hp.loguniform('1-beta2', np.log(1 - np.max(beta2_ar)),
                                             np.log(1 - np.min(beta2_ar)))
        if len(min_lr_ar) == 1:
            space['lr_scheduler_min_lr'] = hp.choice('lr_scheduler_min_lr', min_lr_ar)
        else:
            space['lr_scheduler_min_lr'] = hp.loguniform('lr_scheduler_min_lr',
                                                         np.log(np.min(min_lr_ar)),
                                                         np.log(np.max(min_lr_ar)))
        space['loss_func'] = hp.choice('loss_func', loss_funcs)

        if len(epochs_ar) == 1:
            space['num_epochs'] = hp.choice('num_epochs', epochs_ar)
        else:
            space['num_epochs'] = hp.quniform('num_epochs', np.min(epochs_ar),
                                              np.max(epochs_ar), 10)
        if len(kernal_ar) == 1:
            space['kernel_size'] = hp.choice('kernel_size', kernal_ar)
        else:
            space['kernel_size'] = hp.quniform('kernel_size', np.min(kernal_ar),
                                               np.max(kernal_ar), 1)
        space['max_ch'] = hp.choice('max_ch', max_ch_ar)
        if len(ref_ar) == 1:
            space['norm_ref'] = hp.choice('norm_ref', ref_ar)
        else:
            space['norm_ref'] = hp.loguniform('norm_ref', np.log(np.min(ref_ar)),
                                              np.log(np.max(ref_ar)))
        if len(lambda_ar) == 1:
            space['norm_lambda'] = hp.choice('norm_lambda', lambda_ar)
        else:
            space['norm_lambda'] = hp.uniform('norm_lambda', np.min(lambda_ar),
                                              np.max(lambda_ar))
        space['att_type'] = hp.choice('att_type', att_type_ar1)
        if len(n_head_ar1) == 1:
            space['n_head'] = hp.choice('n_head', n_head_ar1)
        else:
            space['n_head'] = hp.quniform('n_head', np.min(n_head_ar1),
                                          np.max(n_head_ar1), 1)
        if len(drop_prob_ar1) == 1:
            space['drop_prob'] = hp.choice('drop_prob', drop_prob_ar1)
        else:
            space['drop_prob'] = hp.uniform('drop_prob', np.min(drop_prob_ar1),
                                            np.max(drop_prob_ar1))
        if len(n_trans_layers_ar1) == 1:
            space['n_trans_layers'] = hp.choice('n_trans_layers', n_trans_layers_ar1)
        else:
            space['n_trans_layers'] = hp.quniform('n_trans_layers',
                                                  np.min(n_trans_layers_ar1),
                                                  np.max(n_trans_layers_ar1), 1)

        # "hyper-parameters" that doesn't change between tries and we just need to pass them
        space['date_str'] = hp.choice('date_str', [date_str])
        space['count'] = hp.choice('count', [count])
        space['world_size'] = hp.choice('world_size', [world_size])
        space['save_every'] = hp.choice('save_every', [save_every])
        space['model_name'] = hp.choice('model_name', [model_name])

        # Create a Trials object to keep track of evaluations
        trials = Trials()

        # perform hyper-paramter optimization for training
        best = fmin(
            fn=train_objective,  # Objective Function to optimize
            space=space,  # Hyperparameter's Search Space
            algo=tpe.suggest,  # Optimization algorithm (representative TPE)
            max_evals=400,  # Number of optimization attempts
            trials=trials  # Save evaluations in trials
        )

        # save trials
        with open(params.PARAMS_FOLDER + "/" + date_str + '/' + 'trials' + date_str + '.pkl', 'wb') as f:
            pickle.dump(trials, f)
