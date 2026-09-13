"""Three-dimensional Dynamic Snake Convolution."""

import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class DCN_Conv(nn.Module):
    """Sample a deformable line, then convolve across its five points."""

    def __init__(
        self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_conv = nn.Conv3d(in_ch, 3 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm3d(3 * kernel_size)
        self.if_offset = if_offset
        self.morph = morph
        self.extend_scope = extend_scope

        kernels = [
            (1, 1, kernel_size),
            (1, kernel_size, 1),
            (kernel_size, 1, 1),
        ]
        self.dcn_conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=kernels[morph], stride=kernels[morph]
        )
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        offset = torch.tanh(self.bn(self.offset_conv(feature)))
        sampler = DCN(
            feature.shape,
            self.kernel_size,
            self.extend_scope,
            self.morph,
            feature.device,
        )
        deformed_feature = sampler.deform_conv(
            feature, offset, self.if_offset
        )
        return self.relu(self.gn(self.dcn_conv(deformed_feature)))


class DCN:
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.depth = input_shape[2]
        self.width = input_shape[3]
        self.height = input_shape[4]
        self.morph = morph
        self.extend_scope = extend_scope

    @staticmethod
    def _accumulate_from_center(offset):
        """Make each point's displacement depend on points nearer the center."""
        center = offset.shape[1] // 2
        negative = torch.flip(
            torch.cumsum(torch.flip(offset[:, :center], dims=[1]), dim=1),
            dims=[1],
        )
        positive = torch.cumsum(offset[:, center + 1 :], dim=1)
        zero = torch.zeros_like(offset[:, center : center + 1])
        return torch.cat([negative, zero, positive], dim=1)

    def _coordinate_map_3D(self, offset, if_offset):
        batch = offset.shape[0]
        dtype = offset.dtype
        device = offset.device
        z_base, y_base, x_base = torch.meshgrid(
            torch.arange(self.depth, device=device, dtype=dtype),
            torch.arange(self.width, device=device, dtype=dtype),
            torch.arange(self.height, device=device, dtype=dtype),
            indexing="ij",
        )
        z = z_base.expand(batch, self.num_points, -1, -1, -1).clone()
        y = y_base.expand(batch, self.num_points, -1, -1, -1).clone()
        x = x_base.expand(batch, self.num_points, -1, -1, -1).clone()

        center = self.num_points // 2
        spread = torch.arange(
            -center, center + 1, device=device, dtype=dtype
        ).view(1, self.num_points, 1, 1, 1)
        if self.morph == 0:
            x = x + spread
        elif self.morph == 1:
            y = y + spread
        else:
            z = z + spread

        if if_offset:
            z_offset, y_offset, x_offset = torch.split(
                offset, self.num_points, dim=1
            )
            if self.morph != 2:
                z = z + self.extend_scope * self._accumulate_from_center(z_offset)
            if self.morph != 1:
                y = y + self.extend_scope * self._accumulate_from_center(y_offset)
            if self.morph != 0:
                x = x + self.extend_scope * self._accumulate_from_center(x_offset)

        if self.morph == 0:
            order = (0, 2, 3, 4, 1)
            shape = (batch, self.depth, self.width, self.height * self.num_points)
        elif self.morph == 1:
            order = (0, 2, 3, 1, 4)
            shape = (batch, self.depth, self.width * self.num_points, self.height)
        else:
            order = (0, 2, 1, 3, 4)
            shape = (batch, self.depth * self.num_points, self.width, self.height)

        return tuple(axis.permute(order).reshape(shape) for axis in (z, y, x))

    @staticmethod
    def _normalize(coordinate, size):
        if size == 1:
            return torch.zeros_like(coordinate)
        return 2.0 * coordinate / (size - 1) - 1.0

    def _bilinear_interpolate_3D(self, input_feature, z, y, x):
        grid = torch.stack(
            [
                self._normalize(x, self.height),
                self._normalize(y, self.width),
                self._normalize(z, self.depth),
            ],
            dim=-1,
        )
        return F.grid_sample(
            input_feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def deform_conv(self, input_feature, offset, if_offset):
        coordinates = self._coordinate_map_3D(offset, if_offset)
        return self._bilinear_interpolate_3D(input_feature, *coordinates)
