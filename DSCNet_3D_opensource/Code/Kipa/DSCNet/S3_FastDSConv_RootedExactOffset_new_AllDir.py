"""Rooted all-direction 3D Dynamic Snake Convolution."""

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
    """Sample a rooted deformable curve, then convolve across its points."""

    def __init__(
        self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_conv = nn.Conv3d(in_ch, 3 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm3d(3 * kernel_size)
        self.if_offset = if_offset
        self.extend_scope = extend_scope

        self.dcn_conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1, 1),
            stride=(kernel_size, 1, 1),
        )
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        offset = torch.tanh(self.bn(self.offset_conv(feature)))
        sampler = DCN(
            feature.shape,
            self.kernel_size,
            self.extend_scope,
            feature.device,
        )
        deformed_feature = sampler.deform_conv(
            feature, offset, self.if_offset
        )
        return self.relu(self.gn(self.dcn_conv(deformed_feature)))


class DCN:
    def __init__(self, input_shape, kernel_size, extend_scope, device):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        self.num_points = kernel_size
        self.depth = input_shape[2]
        self.width = input_shape[3]
        self.height = input_shape[4]
        self.extend_scope = extend_scope

    @staticmethod
    def _normalize(coordinate, size):
        if size == 1:
            return torch.zeros_like(coordinate)
        return 2.0 * coordinate / (size - 1) - 1.0

    def _sample_vector_field(self, vector_field, z, y, x):
        grid = torch.stack(
            [
                self._normalize(x, self.height),
                self._normalize(y, self.width),
                self._normalize(z, self.depth),
            ],
            dim=-1,
        )
        return F.grid_sample(
            vector_field,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )

    def _clamp_coordinates(self, coordinates):
        limits = (self.depth - 1, self.width - 1, self.height - 1)
        return tuple(
            coordinate.clamp(0, limit)
            for coordinate, limit in zip(coordinates, limits)
        )

    def _coordinate_map_3D(self, offset, if_offset):
        batch = offset.shape[0]
        dtype = offset.dtype
        device = offset.device
        base = torch.meshgrid(
            torch.arange(self.depth, device=device, dtype=dtype),
            torch.arange(self.width, device=device, dtype=dtype),
            torch.arange(self.height, device=device, dtype=dtype),
            indexing="ij",
        )
        base = tuple(axis.expand(batch, -1, -1, -1) for axis in base)

        if not if_offset:
            return tuple(
                axis.unsqueeze(1).expand(-1, self.num_points, -1, -1, -1)
                for axis in base
            )

        offsets = torch.split(
            offset * self.extend_scope, self.num_points, dim=1
        )
        center = self.num_points // 2
        center_field = torch.stack(
            [axis[:, center] for axis in offsets], dim=1
        )
        points = [None] * self.num_points
        points[center] = base

        for direction in (-1, 1):
            adjacent = center + direction
            points[adjacent] = self._clamp_coordinates(
                tuple(
                    base_axis
                    + 0.5 * offset_axis[:, adjacent]
                    + 0.5 * offset_axis[:, center]
                    for base_axis, offset_axis in zip(base, offsets)
                )
            )
            previous = adjacent
            for distance in range(2, center + 1):
                index = center + direction * distance
                sampled_center_offset = self._sample_vector_field(
                    center_field, *points[previous]
                )
                points[index] = self._clamp_coordinates(
                    tuple(
                        previous_axis
                        + 0.5 * offset_axis[:, index]
                        + 0.5 * sampled_center_offset[:, axis_index]
                        for axis_index, (previous_axis, offset_axis) in enumerate(
                            zip(points[previous], offsets)
                        )
                    )
                )
                previous = index

        return tuple(
            torch.stack([point[axis_index] for point in points], dim=1)
            for axis_index in range(3)
        )

    def _vectorized_new_bilinear_interpolate_3D(
        self, input_feature, z, y, x
    ):
        batch, points, depth, width, height = z.shape

        def fold_points(coordinate):
            return coordinate.permute(0, 2, 1, 3, 4).reshape(
                batch, depth * points, width, height
            )

        z, y, x = (fold_points(axis) for axis in (z, y, x))
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
        return self._vectorized_new_bilinear_interpolate_3D(
            input_feature, *coordinates
        )
