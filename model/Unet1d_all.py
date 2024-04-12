import torch
import torch.nn as nn
from torch.nn import init
from . import Transformer
import numpy as np


# https://github.com/LeeJunHyun/Image_Segmentation


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# convolution block module structure with configurable channel input and output dimension and kernel size
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernal_size):
        super(conv_block, self).__init__()
        self.kernal = kernal_size
        self.pad = (self.kernal - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=(self.kernal,),
                      stride=(1,), padding=self.pad, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=(self.kernal,),
                      stride=(1,), padding=self.pad, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# convolution block with up sampling module structure with configurable channel input and output dimension and kernel
# size
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernal_size):
        super(up_conv, self).__init__()
        self.kernal = kernal_size
        self.pad = (self.kernal - 1) // 2
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=(self.kernal,),
                      stride=(1,), padding=self.pad, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# recurrent convolution block module structure with configurable channel output dimension and kernel size and
# recurrent steps
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, kernal_size, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.kernal = kernal_size
        self.pad = (self.kernal - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ch_out, ch_out, kernel_size=(self.kernal,),
                      stride=(1,), padding=self.pad, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


# RCNN block module structure with configurable channel input and output dimension and kernel size and recurrent steps
class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernal_size, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, kernal_size, t=t),
            Recurrent_block(ch_out, kernal_size, t=t)
        )
        self.Conv_1x1 = nn.Conv1d(ch_in, ch_out, kernel_size=(1,), stride=(1,), padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


# Attention block module structure with configurable channel input, intermediate and output dimension
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=(1,), stride=(1,), padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# U-Net model structure with configurable signal channels, output channels, kernel size of convolution layers and
# maximal channel size
class U_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, kernel_size=11, ch_max=128):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.Conv1 = conv_block(ch_in=signal_ch, ch_out=self.mid_out_ch,
                                kernal_size=self.kernal_size)
        self.Conv2 = conv_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv3 = conv_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv4 = conv_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv5 = conv_block(ch_in=8 * self.ch_min, ch_out=self.ch_max,
                                kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Up_conv5 = conv_block(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Up_conv4 = conv_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Up_conv3 = conv_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Up_conv2 = conv_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch,
                                   kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# R2U-Net model structure with RCNN blocks instead of convolution blocks with configurable signal channels,
# output channels, recurrent steps, kernel size of convolution layers and maximal channel size
class R2U_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, t=2, kernel_size=11, ch_max=128):
        super(R2U_Net, self).__init__()

        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=signal_ch, ch_out=self.mid_out_ch, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN2 = RRCNN_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN3 = RRCNN_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN4 = RRCNN_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN5 = RRCNN_block(ch_in=8 * self.ch_min, ch_out=self.ch_max, t=t,
                                  kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Up_RRCNN5 = RRCNN_block(ch_in=self.ch_max, ch_out=8 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Up_RRCNN4 = RRCNN_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Up_RRCNN3 = RRCNN_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Up_RRCNN2 = RRCNN_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch,
                                     t=t, kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# Attention U-Net model structure with Attention blocks after the skip connections of convolution blocks with
# configurable signal channels, output channels, kernel size of convolution layers and maximal channel size
class AttU_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, kernel_size=11, ch_max=128):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.Conv1 = conv_block(ch_in=signal_ch, ch_out=self.mid_out_ch,
                                kernal_size=self.kernal_size)
        self.Conv2 = conv_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv3 = conv_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv4 = conv_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv5 = conv_block(ch_in=8 * self.ch_min, ch_out=self.ch_max,
                                kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Att5 = Attention_block(F_g=8 * self.ch_min, F_l=8 * self.ch_min,
                                    F_int=4 * self.ch_min)
        self.Up_conv5 = conv_block(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Att4 = Attention_block(F_g=4 * self.ch_min, F_l=4 * self.ch_min,
                                    F_int=2 * self.ch_min)
        self.Up_conv4 = conv_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Att3 = Attention_block(F_g=2 * self.ch_min, F_l=2 * self.ch_min,
                                    F_int=self.ch_min)
        self.Up_conv3 = conv_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Att2 = Attention_block(F_g=self.mid_out_ch, F_l=self.mid_out_ch,
                                    F_int=self.mid_out_ch // 2)
        self.Up_conv2 = conv_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch,
                                   kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# Transformer U-Net model structure with customized Transformer blocks after the skip connections of convolution
# blocks with configurable signal channels, output channels, kernel size of convolution layers, maximal channel size
# and Transformer configurations
class TransU_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, kernel_size=11, ch_max=128, n_head=4,
                 drop_prob=0.1, n_layers=1, att_type="Add"):
        super(TransU_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.n_head = n_head
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.att_type = att_type

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        self.mid_out_ch = int(np.ceil((self.mid_out_ch // 2) / self.n_head)
                              * self.n_head * 2)
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.Conv1 = conv_block(ch_in=signal_ch, ch_out=self.mid_out_ch,
                                kernal_size=self.kernal_size)
        self.Conv2 = conv_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv3 = conv_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv4 = conv_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv5 = conv_block(ch_in=8 * self.ch_min, ch_out=self.ch_max,
                                kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans5 = Transformer.Transformer(F_in_enc=8 * self.ch_min,
                                              F_in_dec=8 * self.ch_min,
                                              F_int=4 * self.ch_min,
                                              F_hid=16 * self.ch_min,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_conv5 = conv_block(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans4 = Transformer.Transformer(F_in_enc=4 * self.ch_min,
                                              F_in_dec=4 * self.ch_min,
                                              F_int=2 * self.ch_min,
                                              F_hid=8 * self.ch_min,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_conv4 = conv_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans3 = Transformer.Transformer(F_in_enc=2 * self.ch_min,
                                              F_in_dec=2 * self.ch_min,
                                              F_int=self.ch_min,
                                              F_hid=4 * self.ch_min,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_conv3 = conv_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Trans2 = Transformer.Transformer(F_in_enc=self.mid_out_ch,
                                              F_in_dec=self.mid_out_ch,
                                              F_int=self.mid_out_ch // 2,
                                              F_hid=2 * self.mid_out_ch,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_conv2 = conv_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch,
                                   kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Trans5(enc=d5, dec=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Trans4(enc=d4, dec=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Trans3(enc=d3, dec=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Trans2(enc=d2, dec=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# Transformer U-Net model structure with PyTorch Transformer blocks after the skip connections of convolution
# blocks with configurable signal channels, output channels, kernel size of convolution layers, maximal channel size
# and Transformer configurations
class ClassicTransU_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, kernel_size=11, ch_max=128, n_head=4,
                 drop_prob=0.1, n_layers=1):
        super(ClassicTransU_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.n_head = n_head
        self.drop_prob = drop_prob
        self.n_layers = n_layers

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        self.mid_out_ch = int(np.ceil((self.mid_out_ch // 2) / self.n_head)
                              * self.n_head * 2)
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.Conv1 = conv_block(ch_in=signal_ch, ch_out=self.mid_out_ch,
                                kernal_size=self.kernal_size)
        self.Conv2 = conv_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv3 = conv_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv4 = conv_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min,
                                kernal_size=self.kernal_size)
        self.Conv5 = conv_block(ch_in=8 * self.ch_min, ch_out=self.ch_max,
                                kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans5 = nn.Transformer(d_model=8 * self.ch_min, nhead=self.n_head,
                                     num_encoder_layers=self.n_layers,
                                     num_decoder_layers=self.n_layers,
                                     dim_feedforward=16 * self.ch_min,
                                     dropout=self.drop_prob,
                                     activation=nn.LeakyReLU(inplace=True))

        self.Up_conv5 = conv_block(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans4 = nn.Transformer(d_model=4 * self.ch_min, nhead=self.n_head,
                                     num_encoder_layers=self.n_layers,
                                     num_decoder_layers=self.n_layers,
                                     dim_feedforward=8 * self.ch_min,
                                     dropout=self.drop_prob,
                                     activation=nn.LeakyReLU(inplace=True))
        self.Up_conv4 = conv_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans3 = nn.Transformer(d_model=2 * self.ch_min, nhead=self.n_head,
                                     num_encoder_layers=self.n_layers,
                                     num_decoder_layers=self.n_layers,
                                     dim_feedforward=4 * self.ch_min,
                                     dropout=self.drop_prob,
                                     activation=nn.LeakyReLU(inplace=True))
        self.Up_conv3 = conv_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                                   kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Trans2 = nn.Transformer(d_model=self.mid_out_ch, nhead=self.n_head,
                                     num_encoder_layers=self.n_layers,
                                     num_decoder_layers=self.n_layers,
                                     dim_feedforward=2 * self.mid_out_ch,
                                     dropout=self.drop_prob,
                                     activation=nn.LeakyReLU(inplace=True))
        self.Up_conv2 = conv_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch,
                                   kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Trans5(src=d5.movedim(1, 2), tgt=x4.movedim(1, 2))
        d5 = torch.cat((x4.movedim(1, 2), d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Trans4(src=d4.movedim(1, 2), tgt=x3.movedim(1, 2))
        d4 = torch.cat((x3.movedim(1, 2), d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Trans3(src=d3.movedim(1, 2), tgt=x2.movedim(1, 2))
        d3 = torch.cat((x2.movedim(1, 2), d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Trans2(src=d2.movedim(1, 2), tgt=x1.movedim(1, 2))
        d2 = torch.cat((x1.movedim(1, 2), d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# R2AttU-Net model structure with RCNN blocks instead of convolution blocks, Attention block after the skip
# connections with configurable signal channels, output channels, recurrent steps, kernel size of convolution layers
# and maximal channel size
class R2AttU_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, t=2, kernel_size=11, ch_max=128):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.RRCNN1 = RRCNN_block(ch_in=signal_ch, ch_out=self.mid_out_ch, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN2 = RRCNN_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN3 = RRCNN_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN4 = RRCNN_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN5 = RRCNN_block(ch_in=8 * self.ch_min, ch_out=self.ch_max, t=t,
                                  kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Att5 = Attention_block(F_g=8 * self.ch_min, F_l=8 * self.ch_min,
                                    F_int=4 * self.ch_min)
        self.Up_RRCNN5 = RRCNN_block(ch_in=self.ch_max, ch_out=8 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Att4 = Attention_block(F_g=4 * self.ch_min, F_l=4 * self.ch_min,
                                    F_int=2 * self.ch_min)
        self.Up_RRCNN4 = RRCNN_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Att3 = Attention_block(F_g=2 * self.ch_min, F_l=2 * self.ch_min,
                                    F_int=self.ch_min)
        self.Up_RRCNN3 = RRCNN_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Att2 = Attention_block(F_g=self.mid_out_ch, F_l=self.mid_out_ch,
                                    F_int=self.mid_out_ch // 2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch, t=t,
                                     kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# R2TransU-Net model structure with RCNN blocks instead of convolution blocks, customized Transformer block after the
# skip connections with configurable signal channels, output channels, recurrent steps, kernel size of convolution
# layers and maximal channel size
class R2TransU_Net(nn.Module):
    def __init__(self, signal_ch=1, output_ch=1, t=2, kernel_size=11, ch_max=128, n_head=4,
                 drop_prob=0.1, n_layers=1, att_type="Add"):
        super(R2TransU_Net, self).__init__()

        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.kernal_size = kernel_size
        self.ch_max = ch_max

        self.n_head = n_head
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.att_type = att_type

        self.max_divide = 16
        self.ch_min = self.ch_max // self.max_divide

        self.mid_out_ch = (output_ch + 2 * self.ch_min) // 2
        self.mid_out_ch = int(np.ceil((self.mid_out_ch // 2) / self.n_head)
                              * self.n_head * 2)
        if self.ch_min > output_ch:
            self.mid_out_ch = self.ch_min

        self.RRCNN1 = RRCNN_block(ch_in=signal_ch, ch_out=self.mid_out_ch, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN2 = RRCNN_block(ch_in=self.mid_out_ch, ch_out=2 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN3 = RRCNN_block(ch_in=2 * self.ch_min, ch_out=4 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN4 = RRCNN_block(ch_in=4 * self.ch_min, ch_out=8 * self.ch_min, t=t,
                                  kernal_size=self.kernal_size)

        self.RRCNN5 = RRCNN_block(ch_in=8 * self.ch_min, ch_out=self.ch_max, t=t,
                                  kernal_size=self.kernal_size)

        self.Up5 = up_conv(ch_in=self.ch_max, ch_out=8 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans5 = Transformer.Transformer(F_in_enc=8 * self.ch_min,
                                              F_in_dec=8 * self.ch_min,
                                              F_int=4 * self.ch_min,
                                              F_hid=16 * self.ch_min,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_RRCNN5 = RRCNN_block(ch_in=self.ch_max, ch_out=8 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up4 = up_conv(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans4 = Transformer.Transformer(F_in_enc=4 * self.ch_min,
                                              F_in_dec=4 * self.ch_min,
                                              F_int=2 * self.ch_min,
                                              F_hid=8 * self.ch_min,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_RRCNN4 = RRCNN_block(ch_in=8 * self.ch_min, ch_out=4 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up3 = up_conv(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min,
                           kernal_size=self.kernal_size)
        self.Trans3 = Transformer.Transformer(F_in_enc=2 * self.ch_min,
                                              F_in_dec=2 * self.ch_min,
                                              F_int=self.ch_min,
                                              F_hid=4 * self.ch_min,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_RRCNN3 = RRCNN_block(ch_in=4 * self.ch_min, ch_out=2 * self.ch_min, t=t,
                                     kernal_size=self.kernal_size)

        self.Up2 = up_conv(ch_in=2 * self.ch_min, ch_out=self.mid_out_ch,
                           kernal_size=self.kernal_size)
        self.Trans2 = Transformer.Transformer(F_in_enc=self.mid_out_ch,
                                              F_in_dec=self.mid_out_ch,
                                              F_int=self.mid_out_ch // 2,
                                              F_hid=2 * self.mid_out_ch,
                                              n_head=self.n_head,
                                              drop_prob=self.drop_prob,
                                              att_type=self.att_type,
                                              n_layers=self.n_layers)
        self.Up_RRCNN2 = RRCNN_block(ch_in=2 * self.mid_out_ch, ch_out=self.mid_out_ch,
                                     t=t, kernal_size=self.kernal_size)

        self.Conv_1x1 = nn.Conv1d(self.mid_out_ch, output_ch, kernel_size=(1,),
                                  stride=(1,), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Trans5(enc=d5, dec=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Trans4(enc=d4, dec=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Trans3(enc=d3, dec=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Trans2(enc=d2, dec=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
