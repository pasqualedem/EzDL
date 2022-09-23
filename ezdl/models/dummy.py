import torch
import torch.nn as nn


class Random(nn.Module):
    def __init__(self, arch_params):
        super(Random, self).__init__()
        in_chn, out_chn = arch_params['input_channels'], arch_params['output_channels']

        self.in_chn = in_chn
        self.out_chn = out_chn

    def forward(self, x):
        return torch.rand(x.shape[0], self.out_chn, *(x.shape[2:]), device=x.device)


class SingleConv(nn.Module):
    def __init__(self, arch_params):
        super(SingleConv, self).__init__()
        in_chn, out_chn = arch_params['input_channels'], arch_params['output_channels']
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=3, padding="same")
        self.in_chn = in_chn
        self.out_chn = out_chn

    def forward(self, x):
        return self.conv(x)
