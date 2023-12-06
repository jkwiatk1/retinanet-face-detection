import torch.nn as nn
from typing import List


class FeaturesPyramid(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()

        self.P3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.P4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.P5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.P6 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=3, stride=2, padding=1)
        self.P7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.P5_upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_upsample_2x = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_reduce_aliasing = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.P3_reduce_aliasing = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p6_o = self.P6(c5)
        p7_o = nn.ReLU()(p6_o)
        p7_o = self.P7(p7_o)

        p5_o = self.P5(c5)
        p4_o = self.P4(c4)
        p3_o = self.P3(c3)

        p4_sum = self.P5_upsample_2x(p5_o) + p4_o
        p4_o = self.P4_reduce_aliasing(p4_sum)
        p3_o = self.P3_reduce_aliasing(self.P4_upsample_2x(p4_sum) + p3_o)

        return [p3_o, p4_o, p5_o, p6_o, p7_o]
