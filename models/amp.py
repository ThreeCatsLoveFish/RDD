import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_layer(block, in_planes, out_planes, num_layers, kernel_size=3, stride=1): 
    layers = [block(in_planes, out_planes, kernel_size, stride) for _ in range(num_layers)]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        p = (kernel_size-1)//2
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.pad2 = nn.ReflectionPad2d(p)
        self.conv2 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv1(self.pad1(x)))
        y = self.conv2(self.pad2(y))
        return y + x


class ConvBlock(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=7, stride=1):
        super(ConvBlock, self).__init__()
        p = 3
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv1(self.pad1(x)))


class ConvBlockAfter(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=3, stride=1):
        super(ConvBlockAfter, self).__init__()
        p = 1
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)

    def forward(self, x):
        return self.conv1(self.pad1(x))


class Manipulator(nn.Module):
    def __init__(self, num_resblk, in_planes, output_planes):
        super(Manipulator, self).__init__()
        self.convblks = _make_layer(
            ConvBlock, in_planes, 32, 1, kernel_size=3, stride=1)
        self.convblks_after = _make_layer(
            ConvBlockAfter, 32, 32, 1, kernel_size=3, stride=1)
        self.resblk_raw = _make_layer(
            ResBlock, in_planes, output_planes, num_resblk, kernel_size=3, stride=1)
        self.resblk_diff = _make_layer(
            ResBlock, 32, output_planes, num_resblk, kernel_size=3, stride=1)
        self.amp = nn.Parameter(torch.ones(32, 1))

    def forward(self, x):
        diff = x
        diff = self.convblks(diff)
        diff = self.amp * diff
        diff = self.convblks_after(diff)
        diff = self.resblk_diff(diff)

        return diff


class Amplifier(nn.Module):
    def __init__(self, num_resblk, in_planes, output_planes):
        super(Manipulator, self).__init__()
        self.convblks = _make_layer(
            ConvBlock, in_planes, 32, 1, kernel_size=3, stride=1)
        self.convblks_after = _make_layer(
            ConvBlockAfter, 32, 32, 1, kernel_size=3, stride=1)
        self.resblk_raw = _make_layer(
            ResBlock, in_planes, output_planes, num_resblk, kernel_size=3, stride=1)
        self.resblk_diff = _make_layer(
            ResBlock, 32, output_planes, num_resblk, kernel_size=3, stride=1)
        self.amp = nn.Parameter(torch.ones(32, 1))

    def forward(self, x_a, x_b):
        # extract features
        raw = self.resblk_raw(x_b)

        # amplify the difference
        diff = x_b - x_a
        diff = self.convblks(diff)
        diff = self.amp * diff
        diff = self.convblks_after(diff)
        diff = self.resblk_diff(diff)

        return raw + diff
