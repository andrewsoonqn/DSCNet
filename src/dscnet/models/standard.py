# -*- coding: utf-8 -*-
import torch
from torch import nn, cat

from dscnet.models.dsconv import DCN_Conv


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DSCBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, kernel_size, extend_scope, if_offset, device, conv_cls
    ):
        super(DSCBlock, self).__init__()

        self.conv0 = conv_cls(in_ch, out_ch)
        self.convx = DCN_Conv(
            in_ch, out_ch, kernel_size, extend_scope, 0, if_offset, device
        )
        self.convy = DCN_Conv(
            in_ch, out_ch, kernel_size, extend_scope, 1, if_offset, device
        )
        self.convz = DCN_Conv(
            in_ch, out_ch, kernel_size, extend_scope, 2, if_offset, device
        )
        self.merge = conv_cls(4 * out_ch, out_ch)

    def forward(self, x):
        return self.merge(
            cat(
                [
                    self.conv0(x),
                    self.convx(x),
                    self.convy(x),
                    self.convz(x),
                ],
                dim=1,
            )
        )


class DSCNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernel_size,
        extend_scope,
        if_offset,
        device,
        number,
        dim,
        unet_layers=4,
    ):
        super(DSCNet, self).__init__()
        if unet_layers not in (3, 4, 5):
            raise ValueError("unet_layers must be one of 3, 4, or 5")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")

        self.device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        self.dim = dim
        self.unet_layers = unet_layers

        encoder_channels = [self.number * (2**idx) for idx in range(unet_layers)]

        self.encoder_blocks = nn.ModuleList()
        in_ch = n_channels
        for out_ch in encoder_channels:
            self.encoder_blocks.append(
                DSCBlock(
                    in_ch,
                    out_ch,
                    self.kernel_size,
                    self.extend_scope,
                    self.if_offset,
                    self.device,
                    EncoderConv,
                )
            )
            in_ch = out_ch

        self.decoder_blocks = nn.ModuleList()
        current_ch = encoder_channels[-1]
        for skip_ch in reversed(encoder_channels[:-1]):
            self.decoder_blocks.append(
                DSCBlock(
                    current_ch + skip_ch,
                    skip_ch,
                    self.kernel_size,
                    self.extend_scope,
                    self.if_offset,
                    self.device,
                    DecoderConv,
                )
            )
            current_ch = skip_ch

        self.out_conv = nn.Conv3d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        skip_connections = []

        for level, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
            if level < self.unet_layers - 1:
                x = self.maxpooling(x)

        for block, skip in zip(self.decoder_blocks, reversed(skip_connections[:-1])):
            x = self.up(x)
            x = block(cat([x, skip], dim=1))

        out = self.out_conv(x)
        out = self.softmax(out)
        return out
