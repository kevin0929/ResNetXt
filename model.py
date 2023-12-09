import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return x


class MultiKernelBlock(nn.Module):
    def __init__(self, channels, kernels=[3, 5, 7], stride=1, padding=1):
        super(MultiKernelBlock, self).__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel,
                        stride,
                        padding=kernel // 2,
                        groups=channels,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.ELU(),
                )
                for kernel in kernels
            ]
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return sum(outputs)


class GavelModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=6):
        super(GavelModel, self).__init__()
        self.conv_block = ConvolutionalBlock(
            input_channels, 64, kernel_size=7, stride=2, padding=3
        )
        self.multi_kernel_blocks = nn.Sequential(
            MultiKernelBlock(64), MultiKernelBlock(64), MultiKernelBlock(64)
        )
        self.final_conv = nn.Conv1d(
            64, 128, kernel_size=1
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.multi_kernel_blocks(x)
        x = self.final_conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
