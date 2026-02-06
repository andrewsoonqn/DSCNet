   # -*- coding: utf-8 -*-
import torch
from torch import nn, cat


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch) # Separates out_ch number of channels into out_ch // 4 number of groups (Each group has 4 channels), normalizes within group
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): 
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DCN_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device):
        super(DCN_Conv, self).__init__()
        # hardcoded for kernel_size = 5
        self.kernel_size = 7
        self.offset_conv3 = nn.Conv3d(in_ch, 3, 3, padding=1)
        self.offset_conv5 = nn.Conv3d(in_ch, 3, 5, padding=2)
        self.offset_conv7 = nn.Conv3d(in_ch, 3, 7, padding=3)
        self.bn = nn.BatchNorm3d(3 * self.kernel_size) # Normalizes across the Channel dimension; returns same shape as input
        self.device = device

        self.if_offset = if_offset
        self.morph = morph
        self.extend_scope = extend_scope

        self.dcn_conv_x = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, self.kernel_size), stride=(1, 1, self.kernel_size), padding=0)  # conv in x-direction
        self.dcn_conv_y = nn.Conv3d(in_ch, out_ch, kernel_size=(1, self.kernel_size, 1), stride=(1, self.kernel_size, 1), padding=0)  # conv in y-direction
        self.dcn_conv_z = nn.Conv3d(in_ch, out_ch, kernel_size=(self.kernel_size, 1, 1), stride=(self.kernel_size, 1, 1), padding=0)  # conv in z-direction

        #self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f):
        # Input: [N, C, D, W, H];
        offset0 = torch.zeros(f.size(dim=0), 3, f.size(dim=2), f.size(dim=3), f.size(dim=4))
        offset0 = offset0.to(self.device)
        offset3 = self.offset_conv3(f)
        offset5 = self.offset_conv5(f)
        offset7 = self.offset_conv7(f)
        offset = torch.concat([offset7, offset5, offset3, offset0, offset3, offset5, offset7], dim=1) # Output: [N, 3*C, D, W, H]
        offset = self.bn(offset) # Output: [N, 3*C, D, W, H];
        offset = torch.tanh(offset) # Output: [N, 3*C, D, W, H]; tanh is (-1, 1)
        #offset = offset.mul(self.extend_scope)
        input_shape = f.shape # shape: [N, C, D, W, H];

        dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)
        deformed_feature = dcn.deform_conv(f, offset, self.if_offset) # _coordinate_map_3D (Output: [N, D, W, H*C] OR [N, D, W*C, H] OR [N, D*C, W, H]) + _bilinear_interpolate_3D (Output: [N, C, D, W, H])

        # Only ever does one of the following
        if self.morph == 0:
            x = self.dcn_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        elif self.morph == 1:
            x = self.dcn_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dcn_conv_z(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)  
            return x

class DCN(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.depth = input_shape[2]
        self.width = input_shape[3]
        self.height = input_shape[4]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
        self.num_batch = input_shape[0]   # (N,C,D,W,H)
        self.num_channels = input_shape[1]

    '''
    input: offset [N,3*K,D,W,H]
    output: [N,1,K*D,W,H]   coordinate map
    output: [N,1,K,K*W,H]   coordinate map
    output: [N,1,D,W,K*H]   coordinate map
    '''
    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        #offset1, offset2 = torch.split(offset, 3 * self.num_points, dim=1) # Split offset into groups of 3*self.num_points i.e. [N, 3*C, D, W, H]
        z_offset1, y_offset1, x_offset1 = torch.split(offset, self.num_points, dim=1) # Split offset1 into groups of self.num_points i.e. [N, C, D, W, H]
        #z_offset2, y_offset2, x_offset2 = torch.split(offset2, self.num_points, dim=1) # Split offset2 into groups of self.num_points i.e. [N, C, D, W, H]

        z_center = torch.arange(0, self.depth).repeat([self.width*self.height]) # [0 to self.depth] * self.width * self.height = [D*W*H]
        z_center = z_center.reshape(self.width, self.height, self.depth) # [W, H, D]
        z_center = z_center.permute(2, 1, 0) # [D, W, H]
        z_center = z_center.reshape([-1, self.depth, self.width, self.height]) # [1, D, W, H]
        z_center = z_center.repeat([self.num_points, 1, 1, 1]).float() # [C, D, W, H]
        z_center = z_center.unsqueeze(0) # [1, C, D, W, H] with running numbers going in depth direction

        y_center = torch.arange(0, self.width).repeat([self.height * self.depth]) # [0 to self.width] * self.height * self.depth] = [W*H*D]
        y_center = y_center.reshape(self.height, self.depth, self.width) # [H, D, W]
        y_center = y_center.permute(1, 2, 0) # [D, W, H]
        y_center = y_center.reshape([-1, self.depth, self.width, self.height]) # [1, D, W, H]
        y_center = y_center.repeat([self.num_points, 1, 1, 1]).float() # [C, D, W, H]
        y_center = y_center.unsqueeze(0) # [1, C, D, W, H] with running numbers going in width direction

        x_center = torch.arange(0, self.height).repeat([self.depth * self.width]) # [0 to self.height] * self.depth * self.width] = [H*D*W]
        x_center = x_center.reshape(self.depth, self.width, self.height) # [D, W, H]
        x_center = x_center.permute(0, 1, 2) # [D, W, H] (not necessary?)
        x_center = x_center.reshape([-1, self.depth, self.width, self.height]) # [1, D, W, H]
        x_center = x_center.repeat([self.num_points, 1, 1, 1]).float() # [C, D, W, H]
        x_center = x_center.unsqueeze(0) # [1, C, D, W, H] with running numbers going in height direction

        if self.morph == 0:
            z = torch.linspace(0, 0, 1) # start=0, end=0, size=1, [1]
            y = torch.linspace(0, 0, 1) # start=0, end=0, size=1, [1]
            x = torch.linspace(-int(self.num_points//2), int(self.num_points//2), int(self.num_points)) # start=-self.num_points//2, end=self.num_points//2, size=self.num_points, [C]
            z, y, x = torch.meshgrid(z, y, x) # z = [1, 1, C] all zeros, y = [1, 1, C] all zeros, x = [1, 1, C] -self.num_points//2 to self.num_points//2
            z_spread = z.reshape(-1, 1) # z = [C, 1] all zeros (-1 means infer that dimension)
            y_spread = y.reshape(-1, 1) # y = [C, 1] all zeros
            x_spread = x.reshape(-1, 1) # x = [C, 1] -self.num_points//2 to self.num_points//2

            z_grid = z_spread.repeat([1, self.depth * self.width * self.height]) # [C, D*W*H] all zeros
            z_grid = z_grid.reshape([self.num_points, self.depth, self.width, self.height]) # [C, D, W, H]
            z_grid = z_grid.unsqueeze(0)  # [1, C, D, W, H] all zeros

            y_grid = y_spread.repeat([1, self.depth * self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.depth, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [1, C, D, W, H] all zeros

            x_grid = x_spread.repeat([1, self.depth * self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.depth, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [1, C, D, W, H] C running from -self.num_points//2 to self.num_points//2

            z_new = z_center + z_grid # [1, C, D, W, H] 0 to depth in the D axis
            y_new = y_center + y_grid # [1, C, D, W, H] 0 to width in the W axis
            x_new = x_center + x_grid # [1, C, D, W, H] 0 to height in the H axis and -self.num_points//2 to self.num_points//2 in the C axis

            z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1) # [N, C, D, W, H] 0 to depth in the D axis
            y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1) # [N, C, D, W, H] 0 to width in the W axis
            x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1) # [N, C, D, W, H] 0 to height in the H axis and -self.num_points//2 to self.num_points//2 in the C axis

            z_new = z_new.to(self.device)
            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device) # send matrix to device

            z_offset1_new = z_offset1.detach().clone() # detach ensures that gradients do not propagate from z_offset1_new to z_offset1; clone creates an independent matrix in memory
            y_offset1_new = y_offset1.detach().clone() # detach ensures that gradients do not propagate from y_offset1_new to y_offset1; clone creates an independent matrix in memory

            if if_offset:
                z_offset1_new = z_offset1_new.permute(1, 0, 2, 3, 4) # [C, N, D, W, H] why? offset in channel direction.
                y_offset1_new = y_offset1_new.permute(1, 0, 2, 3, 4) # [C, N, D, W, H]
                z_offset1 = z_offset1.permute(1, 0, 2, 3, 4) # [C, N, D, W, H]
                y_offset1 = y_offset1.permute(1, 0, 2, 3, 4) # [C, N, D, W, H]
                center = int(self.num_points // 2) # 2
                #z_offset1_new[center] = 0
                #y_offset1_new[center] = 0
                for index in range(1, center + 1): # 1, 2 (kernel is 0, 1, center, 3, 4 -> size 5)
                    z_offset1_new[center + index] = z_offset1_new[center + index - 1] + z_offset1[center + index] # next offset is dependent on previous
                    z_offset1_new[center - index] = z_offset1_new[center - index + 1] + z_offset1[center - index]
                    y_offset1_new[center + index] = y_offset1_new[center + index - 1] + y_offset1[center + index]
                    y_offset1_new[center - index] = y_offset1_new[center - index + 1] + y_offset1[center - index]
                z_offset1_new = z_offset1_new.permute(1, 0, 2, 3, 4).to(self.device) # [N, C, D, W, H]
                y_offset1_new = y_offset1_new.permute(1, 0, 2, 3, 4).to(self.device) # [N, C, D, W, H]
                z_new = z_new.add(z_offset1_new.mul(self.extend_scope)) # multiply new offsets by self.extend_scope, then add z_offset1_new to z_new (which is all zeros except for depth)
                y_new = y_new.add(y_offset1_new.mul(self.extend_scope)) # multiply new offsets by self.extend_scope, then add y_offset1_new to y_new (which is all zeros except for width)

                z_new = z_new.reshape([self.num_batch, 1, 1, self.num_points, self.depth, self.width, self.height]) # [N, 1, 1, C, D, W, H]
                z_new = z_new.permute(0, 4, 1, 5, 2, 6, 3) # [N, D, 1, W, 1, H, C]
                z_new = z_new.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height]) # [N, D, W, H*C]

                y_new = y_new.reshape([self.num_batch, 1, 1, self.num_points, self.depth, self.width, self.height])
                y_new = y_new.permute(0, 4, 1, 5, 2, 6, 3)
                y_new = y_new.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height]) # [N, D, W, H*C]

                x_new = x_new.reshape([self.num_batch, 1, 1, self.num_points, self.depth, self.width, self.height])
                x_new = x_new.permute(0, 4, 1, 5, 2, 6, 3)
                x_new = x_new.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height]) # [N, D, W, H*C]
            return z_new, y_new, x_new

        elif self.morph == 1:
            z = torch.linspace(0, 0, 1)
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
            x = torch.linspace(0, 0, 1)
            z, y, x = torch.meshgrid(z, y, x)
            z_spread = z.reshape(-1, 1)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            z_grid = z_spread.repeat([1, self.depth * self.width * self.height])
            z_grid = z_grid.reshape([self.num_points, self.depth, self.width, self.height])
            z_grid = z_grid.unsqueeze(0)  # [N*K,D,W,H]

            y_grid = y_spread.repeat([1, self.depth * self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.depth, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            x_grid = x_spread.repeat([1, self.depth * self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.depth, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            z_new = z_center + z_grid
            y_new = y_center + y_grid
            x_new = x_center + x_grid  # [N*K*K*K,D,W,H]

            z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1)
            y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1)

            z_new = z_new.to(self.device)
            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset1_new = x_offset1.detach().clone()
            z_offset1_new = z_offset1.detach().clone()

            if if_offset:
                x_offset1_new = x_offset1_new.permute(1, 0, 2, 3, 4)
                z_offset1_new = z_offset1_new.permute(1, 0, 2, 3, 4)
                x_offset1 = x_offset1.permute(1, 0, 2, 3, 4)
                z_offset1 = z_offset1.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)
                #x_offset1_new[center] = 0
                #z_offset1_new[center] = 0
                for index in range(1, center + 1):
                    x_offset1_new[center + index] = x_offset1_new[center + index - 1] + x_offset1[center + index]
                    x_offset1_new[center - index] = x_offset1_new[center - index + 1] + x_offset1[center - index]
                    z_offset1_new[center + index] = z_offset1_new[center + index - 1] + z_offset1[center + index]
                    z_offset1_new[center - index] = z_offset1_new[center - index + 1] + z_offset1[center - index]
                x_offset1_new = x_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                z_offset1_new = z_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                z_new = z_new.add(z_offset1_new.mul(self.extend_scope))
                x_new = x_new.add(x_offset1_new.mul(self.extend_scope))
            z_new = z_new.reshape([self.num_batch, 1, self.num_points, 1, self.depth, self.width, self.height])
            z_new = z_new.permute(0, 4, 1, 5, 2, 6, 3)
            z_new = z_new.reshape([self.num_batch, self.depth, self.num_points * self.width, self.height])
            y_new = y_new.reshape([self.num_batch, 1, self.num_points, 1, self.depth, self.width, self.height])
            y_new = y_new.permute(0, 4, 1, 5, 2, 6, 3)
            y_new = y_new.reshape([self.num_batch, self.depth, self.num_points * self.width, self.height])
            x_new = x_new.reshape([self.num_batch, 1, self.num_points, 1, self.depth, self.width, self.height])
            x_new = x_new.permute(0, 4, 1, 5, 2, 6, 3)
            x_new = x_new.reshape([self.num_batch, self.depth, self.num_points * self.width, self.height])
            return z_new, y_new, x_new

        else:
            z = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(0, 0, 1)
            z, y, x = torch.meshgrid(z, y, x)
            z_spread = z.reshape(-1, 1)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            z_grid = z_spread.repeat([1, self.depth * self.width * self.height])
            z_grid = z_grid.reshape([self.num_points, self.depth, self.width, self.height])
            z_grid = z_grid.unsqueeze(0)  # [N*K,D,W,H]

            y_grid = y_spread.repeat([1, self.depth * self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.depth, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            x_grid = x_spread.repeat([1, self.depth * self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.depth, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            z_new = z_center + z_grid
            y_new = y_center + y_grid
            x_new = x_center + x_grid  # [N*K*K*K,D,W,H]

            z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1)
            y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1)
            
            z_new = z_new.to(self.device)
            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset1_new = x_offset1.detach().clone()
            y_offset1_new = y_offset1.detach().clone()

            if if_offset:
                x_offset1_new = x_offset1_new.permute(1, 0, 2, 3, 4)
                y_offset1_new = y_offset1_new.permute(1, 0, 2, 3, 4)
                x_offset1 = x_offset1.permute(1, 0, 2, 3, 4)
                y_offset1 = y_offset1.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)
                #x_offset1_new[center] = 0
                #y_offset1_new[center] = 0
                for index in range(1, center + 1):
                    x_offset1_new[center + index] = x_offset1_new[center + index - 1] + x_offset1[center + index]
                    x_offset1_new[center - index] = x_offset1_new[center - index + 1] + x_offset1[center - index]
                    y_offset1_new[center + index] = y_offset1_new[center + index - 1] + y_offset1[center + index]
                    y_offset1_new[center - index] = y_offset1_new[center - index + 1] + y_offset1[center - index]
                x_offset1_new = x_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                y_offset1_new = y_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                x_new = x_new.add(x_offset1_new.mul(self.extend_scope))
                y_new = y_new.add(y_offset1_new.mul(self.extend_scope))

            z_new = z_new.reshape([self.num_batch, self.num_points, 1, 1, self.depth, self.width, self.height])
            z_new = z_new.permute(0, 4, 1, 5, 2, 6, 3)
            z_new = z_new.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height])

            y_new = y_new.reshape([self.num_batch, self.num_points, 1, 1, self.depth, self.width, self.height])
            y_new = y_new.permute(0, 4, 1, 5, 2, 6, 3)
            y_new = y_new.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height])

            x_new = x_new.reshape([self.num_batch, self.num_points, 1, 1, self.depth, self.width, self.height])
            x_new = x_new.permute(0, 4, 1, 5, 2, 6, 3)
            x_new = x_new.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height])
            return z_new, y_new, x_new

    '''
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    '''
    def _bilinear_interpolate_3D(self, input_feature, z, y, x):
        z = z.reshape([-1]).float() # [N*D*W*H*C]
        y = y.reshape([-1]).float() # [N*D*W*H*C]
        x = x.reshape([-1]).float() # [N*D*W*H*C]

        zero = torch.zeros([]).int()
        max_z = self.depth - 1
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        z0 = torch.floor(z).int() # returns a new tensor with each element replaced by the largest integer smaller than it
        z1 = z0 + 1 # z0 is the lower integer limit and z1 is the upper integer limit
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume (this is for preventing illegal indexing)
        z0 = torch.clamp(z0, zero, max_z) # every element less than zero is replaced with zero and every element greater than max_z is replaced with max_z
        z1 = torch.clamp(z1, zero, max_z)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)  # [N*D*W*H*C]

        # convert input_feature and coordinate X, Y to 3D，for gathering
        # input_feature_flat = input_feature.reshape([-1, self.num_channels])   # [N*D*W*H, C]
        input_feature_flat = input_feature.flatten() # [N*C*D*W*H]
        input_feature_flat = input_feature_flat.reshape(self.num_batch, self.num_channels, self.depth, self.width,
                                                        self.height) # [N, C, D, W, H] why is this necessary?
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 4, 1) # [N, D, W, H, C] 
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels) # [N*D*W*H, C]
        dimension = self.height * self.width * self.depth

        base = torch.arange(self.num_batch) * dimension # [N] why multiply by dimension?
        base = base.reshape([-1, 1]).float()  # [N,1]

        repeat = torch.ones([self.num_points * self.depth * self.width * self.height]).unsqueeze(0) # unsqueeze returns a new tensor with a dimension of 1 inserted in the indicated location; [1, N*D*W*H]
        repeat = repeat.float()  # [1, N*D*W*H]

        base = torch.matmul(base, repeat)  # [N, 1] * [1, N*D*W*H]  ==> [N, N*D*W*H]
        base = base.reshape([-1])  # [N*N*D*W*H] what is this doing?

        base = base.to(self.device) # what is the point of this? torch.arange and torch.repeat are done on CPU by default. Have to move to same device as the other tensors.

        base_z0 = base + z0 * self.height * self.width
        base_z1 = base + z1 * self.height * self.width
        base_y0 = base + y0 * self.height # why no width? because the reshaping flattened everything into one axis. To properly get the index, you have to multiply differently.
        base_y1 = base + y1 * self.height 

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 + base_z0 - base + x0
        index_b0 = base_y0 + base_z1 - base + x0
        index_c0 = base_y0 + base_z0 - base + x1
        index_d0 = base_y0 + base_z1 - base + x1  # [N*KD*KW*KH]

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 + base_z0 - base + x0
        index_b1 = base_y1 + base_z1 - base + x0
        index_c1 = base_y1 + base_z0 - base + x1
        index_d1 = base_y1 + base_z1 - base + x1  # [N*KD*KW*KH]

        # get 8 grid values  ([N*D*W*H,C], [N*D*W*H*27])
        value_a0 = input_feature_flat[index_a0.type(torch.int64)]
        value_b0 = input_feature_flat[index_b0.type(torch.int64)]
        value_c0 = input_feature_flat[index_c0.type(torch.int64)]
        value_d0 = input_feature_flat[index_d0.type(torch.int64)]
        value_a1 = input_feature_flat[index_a1.type(torch.int64)]
        value_b1 = input_feature_flat[index_b1.type(torch.int64)]
        value_c1 = input_feature_flat[index_c1.type(torch.int64)]
        value_d1 = input_feature_flat[index_d1.type(torch.int64)]  # [N*KD*KW*KH, C]

        # find 8 grid locations
        z0 = torch.floor(z).int()
        z1 = z0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume (this is for calculating weights)
        z0 = torch.clamp(z0, zero, max_z + 1) 
        z1 = torch.clamp(z1, zero, max_z + 1)
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)  # [N*KD*KW*KH]

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        z0_float = z0.float()
        z1_float = z1.float()

        vol_a0 = ((z1_float - z) * (y1_float - y) * (x1_float - x)).unsqueeze(-1)  
        vol_b0 = ((z - z0_float) * (y1_float - y) * (x1_float - x)).unsqueeze(-1)
        vol_c0 = ((z1_float - z) * (y1_float - y) * (x - x0_float)).unsqueeze(-1)
        vol_d0 = ((z - z0_float) * (y1_float - y) * (x - x0_float)).unsqueeze(-1)
        vol_a1 = ((z1_float - z) * (y - y0_float) * (x1_float - x)).unsqueeze(-1)
        vol_b1 = ((z - z0_float) * (y - y0_float) * (x1_float - x)).unsqueeze(-1)
        vol_c1 = ((z1_float - z) * (y - y0_float) * (x - x0_float)).unsqueeze(-1)
        vol_d1 = ((z - z0_float) * (y - y0_float) * (x - x0_float)).unsqueeze(-1)  # [N*KD*KW*KH, C]

        outputs = value_a0 * vol_a0 + value_b0 * vol_b0 + value_c0 * vol_c0 + value_d0 * vol_d0 + value_a1 * vol_a1 + value_b1 * vol_b1 + value_c1 * vol_c1 + value_d1 * vol_d1

        if self.morph == 0:
            outputs = outputs.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        elif self.morph == 1:
            outputs = outputs.reshape([self.num_batch, self.depth, self.num_points*self.width, self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        else:
            outputs = outputs.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        z, y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, z, y, x)
        return deformed_feature
