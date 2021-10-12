import torch
import torch.nn as nn
from collections import OrderedDict


class DownBlock(nn.Module):
    r"""
    Down-sampling block of the model. It could be instantiated with
    a 2x2 Conv2D subsampling or 1x1 Conv1D + max-pooling subsampling.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        groups (in): number of groups in the case of Conv2D.
        mode (str): defines the type of down-sampling, if mode=Conv1, the down-sampling
            is 1x1 Conv2D + max-pooling, else, the down-sampling is a 2x2 Conv2D.
    """
    def __init__(self, in_channels, out_channels, groups, mode='Conv2'):
        super(DownBlock, self).__init__()

        if mode == 'Conv1':
            self.down_sample = nn.Sequential(OrderedDict([
                ('conv',  nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                ('pool', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('act', nn.PReLU(out_channels))
            ]))
        else:
            self.down_sample = nn.Sequential(OrderedDict([
                ('conv',  nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, groups=groups)),
                ('act', nn.PReLU(out_channels))
            ]))

    def forward(self, x):
        return self.down_sample(x)


class UpBlock(nn.Module):
    r"""
    Up-sampling block of the model.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        groups (in): number of groups.
    """
    def __init__(self, in_channels, cat_channels, out_channels, groups=16):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2, groups=groups)
        self.act = nn.PReLU(out_channels)
        self.act_t = nn.PReLU(in_channels)

    def forward(self, x):
        up, concat = x
        up = self.act_t(self.conv_t(up))

        return self.act(self.conv(torch.cat([concat, up], 1)))


class InputBlock(nn.Module):
    r"""
    Input block of the model.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()

        self.input_block = nn.Sequential(OrderedDict([
            ('conv 1', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            ('act 1', nn.PReLU(out_channels)),
        ]))

    def forward(self, x):
        return self.input_block(x)


class OutputBlock(nn.Module):
    r"""
    Output block of the model.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()

        self.output_block = nn.Sequential(OrderedDict([
            ('conv 1', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            ('act 1', nn.PReLU(out_channels)),
        ]))

    def forward(self, x):
        return self.output_block(x)


class DenoisingBlock(nn.Module):
    r"""
    Denoising block of the model.

    Args:
        in_channels (int): number of input channels of the block.
        inner_channels (int): number of channels of the inner (dense) convolutions.
        out_channels (int): number of output channels of the block.
        inner_convolutions (int): number of inner/dense convolutions.
        groups (int): number groups of convolutions.
    """
    def __init__(self, in_channels, inner_channels, out_channels, inner_convolutions=1, groups=16):
        super(DenoisingBlock, self).__init__()

        self.inner_convolutions = inner_convolutions
        self.residual = True if in_channels == out_channels else False

        self.input_convolution = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, inner_channels, kernel_size=1)),
            ('act', nn.PReLU(inner_channels))
        ]))

        self.dense_convolutions = nn.ModuleList()
        for i in range(1, inner_convolutions + 1):
            dense_convolution = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels + i * inner_channels,
                                   inner_channels,
                                   kernel_size=3,
                                   padding=1,
                                   groups=groups)),
                ('act', nn.PReLU(inner_channels))
            ]))

            self.dense_convolutions.append(dense_convolution)

        self.output_convolution = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=in_channels + (inner_convolutions + 1) * inner_channels,
                               out_channels=out_channels,
                               kernel_size=1)),
            ('act', nn.PReLU(out_channels))
        ]))

    def forward(self, x):
        output = self.input_convolution(x)
        output = torch.cat([x, output], 1)

        for i in range(self.inner_convolutions):
            inner_output = self.dense_convolutions[i](output)
            output = torch.cat([output, inner_output], 1)

        output = self.output_convolution(output)

        if self.residual:
            output = output + x

        return output


class LightRDUNet(nn.Module):
    r"""
    LRDUNet model
    """
    def __init__(self, **kwargs):
        super().__init__()

        groups = kwargs['groups']
        downsampling = kwargs['downsampling']
        dense_convolutions = kwargs['dense convolutions']
        self.residual = kwargs['residual']

        filters_0 = kwargs['base filters']
        filters_1 = 2 * kwargs['base filters']
        filters_2 = 4 * kwargs['base filters']
        filters_3 = 8 * kwargs['base filters']

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(1, filters_0)
        self.block_0_0 = nn.Sequential(OrderedDict([
            ('encode 0 0', DenoisingBlock(filters_0, filters_0 // 2, filters_0, dense_convolutions, groups)),
            ('encode 0 1', DenoisingBlock(filters_0, filters_0 // 2, filters_0, dense_convolutions, groups))
        ]))

        self.down_0 = DownBlock(filters_0, filters_1, groups, downsampling)

        # Level 1:
        self.block_1_0 = nn.Sequential(OrderedDict([
            ('encode 1 0', DenoisingBlock(filters_1, filters_1 // 2, filters_1, dense_convolutions, groups)),
            ('encode 1 1', DenoisingBlock(filters_1, filters_1 // 2, filters_1, dense_convolutions, groups))
        ]))

        self.down_1 = DownBlock(filters_1, filters_2, groups, downsampling)

        # Level 2:
        self.block_2_0 = nn.Sequential(OrderedDict([
            ('encode 2 0', DenoisingBlock(filters_2, filters_2 // 2, filters_2, dense_convolutions)),
            ('encode 2 1', DenoisingBlock(filters_2, filters_2 // 2, filters_2, dense_convolutions))
        ]))

        self.down_2 = DownBlock(filters_2, filters_3, groups, downsampling)

        # Bottleneck
        # Level 3:
        self.block_3_0 = nn.Sequential(OrderedDict([
            ('bottleneck 0', DenoisingBlock(filters_3, filters_3 // 2, filters_3, dense_convolutions)),
            ('bottleneck 1', DenoisingBlock(filters_3, filters_3 // 2, filters_3, dense_convolutions))
        ]))

        # Level 2:
        self.up_2 = UpBlock(filters_3, filters_2, filters_2, groups)
        self.block_2_1 = nn.Sequential(OrderedDict([
            ('decode 2 0 ', DenoisingBlock(filters_2, filters_2 // 2, filters_2, dense_convolutions, groups)),
            ('decode 2 1 ', DenoisingBlock(filters_2, filters_2 // 2, filters_2, dense_convolutions, groups))
        ]))

        # Level 1:
        self.up_1 = UpBlock(filters_2, filters_1, filters_1, groups)
        self.block_1_1 = nn.Sequential(OrderedDict([
            ('decode 1 0', DenoisingBlock(filters_1, filters_1 // 2, filters_1, dense_convolutions, groups)),
            ('decode 1 1', DenoisingBlock(filters_1, filters_1 // 2, filters_1, dense_convolutions, groups))
        ]))

        # Level 0:
        self.up_0 = UpBlock(filters_1, filters_0, filters_0, groups)
        self.block_0_1 = nn.Sequential(OrderedDict([
            ('decode 0 0', DenoisingBlock(filters_0, filters_0 // 2, filters_0, dense_convolutions, groups)),
            ('decode 0 1', DenoisingBlock(filters_0, filters_0 // 2, filters_0, dense_convolutions, groups))
        ]))

        self.output_block = OutputBlock(filters_0, 1)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)
        out_0 = self.block_0_0(out_0)       # Level 0

        out_1 = self.down_0(out_0)
        out_1 = self.block_1_0(out_1)       # Level 1

        out_2 = self.down_1(out_1)
        out_2 = self.block_2_0(out_2)       # Level 2

        out_3 = self.down_2(out_2)
        out_3 = self.block_3_0(out_3)       # Level 3 (Bottleneck)

        out_2 = self.up_2([out_3, out_2])
        out_2 = self.block_2_1(out_2)       # Level 2

        out_1 = self.up_1([out_2, out_1])
        out_1 = self.block_1_1(out_1)       # Level 1

        out_0 = self.up_0([out_1, out_0])
        out_0 = self.block_0_1(out_0)       # Level 0

        x = self.output_block(out_0)

        if self.residual:
            x = x + inputs

        return x
