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
        self.kernel_size = 5
        self.offset_conv = nn.Conv3d(in_ch, 2 * in_ch * self.kernel_size, 3, padding=1) # takes in [N, C, D, W, H] and returns [N, 2*C*K, D, W, H]
        self.bn = nn.BatchNorm3d(2 * in_ch * self.kernel_size) # Normalizes across the channel dimension;
        self.device = device

        self.if_offset = if_offset
        self.morph = morph
        self.extend_scope = extend_scope

        self.dcn_conv_x = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, self.kernel_size), stride=(1, 1, self.kernel_size), padding=0)  # conv in x-direction; takes in [N, C_in, D, W, K*H], and returns [N, C_out, D, W, H]
        self.dcn_conv_y = nn.Conv3d(in_ch, out_ch, kernel_size=(1, self.kernel_size, 1), stride=(1, self.kernel_size, 1), padding=0)  # conv in y-direction; takes in [N, C_in, D, K*W, H], and returns [N, C_out, D, W, H]
        self.dcn_conv_z = nn.Conv3d(in_ch, out_ch, kernel_size=(self.kernel_size, 1, 1), stride=(self.kernel_size, 1, 1), padding=0)  # conv in z-direction; takes in [N, C_in, K*D, W, H], and returns [N, C_out, D, W, H]

        #self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.dcn = None


    def forward(self, f):
        # Input: [N, K, D, W, H];
        offset = self.offset_conv(f) # Output: [N, C*2*K, D, W, H];
        offset = self.bn(offset) # Output: [N, C*2*K, D, W, H];
        offset = torch.tanh(offset) # Output: [N, C*2*K, D, W, H]; tanh is (-1, 1)
        input_shape = f.shape # shape: [N, C, D, W, H];

        if self.dcn is None:
            self.dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)

        deformed_feature = self.dcn.deform_conv(f, offset, self.if_offset) # _coordinate_map_3D (Output: [N, C, K, D, W, H] ) + _vectorized_new_bilinear_interpolate_3D (Output: [N, C, D, W, H*K] OR [N, C, D, W*K, H] OR [N, C, D*K, W, H])

        # Only ever does one of the following
        if self.morph == 0:
            x = self.dcn_conv_x(deformed_feature) # [N, C_out, D, W, H]
            x = self.gn(x)
            x = self.relu(x)
            return x
        elif self.morph == 1:
            x = self.dcn_conv_y(deformed_feature) # [N, C_out, D, W, H]
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dcn_conv_z(deformed_feature) # [N, C_out, D, W, H]
            x = self.gn(x)
            x = self.relu(x)  
            return x

class DCN(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size # K
        self.depth = input_shape[2] # D
        self.width = input_shape[3] # W
        self.height = input_shape[4] # H
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
        self.num_channels = input_shape[1] # C

    '''
    input: offset [N,2*C*K,D,W,H]
    output: [N,C,K,D,W,H]   coordinate map
    '''
    def _coordinate_map_3D(self, offset, if_offset):
        # variable num_batch
        num_batch = offset.shape[0] # N

        # offset
        offset1, offset2 = torch.split(offset, self.num_channels*self.num_points, dim=1) # Split offset1 into groups of self.num_channels*self.num_points i.e. [N, C*K, D, W, H]
        offset1 = offset1.reshape(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height)
        offset2 = offset2.reshape(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height)
        
        z_coords = torch.arange(self.depth) # [D]
        y_coords = torch.arange(self.width) # [W]
        x_coords = torch.arange(self.height) # [H]
        z_center, y_center, x_center = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij') # [D, W, H] where z increases in depth direction, y in width, and x in height.
        z_center = z_center.view(1, 1, 1, self.depth, self.width, self.height) # [1, 1, 1, D, W, H]
        y_center = y_center.view(1, 1, 1, self.depth, self.width, self.height) # [1, 1, 1, D, W, H]
        x_center = x_center.view(1, 1, 1, self.depth, self.width, self.height) # [1, 1, 1, D, W, H]
        z_center = z_center.expand(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height).float() # [N, C, K, D, W, H]
        y_center = y_center.expand(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height).float() # [N, C, K, D, W, H]
        x_center = x_center.expand(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height).float() # [N, C, K, D, W, H]

        if self.morph == 0:
            x_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
            x_spread = x_spread.view(1, 1, self.num_points, 1, 1, 1) # [1, 1, K, 1, 1, 1]
            x_grid = x_spread.expand(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height) # (N, C, K, D, W, H)

            z_new = z_center.to(self.device)
            y_new = y_center.to(self.device)
            x_new = (x_center + x_grid).to(self.device)

            z_offset = offset1.detach().clone() # detach ensures that gradients do not propagate from z_offset to offset1; clone creates an independent matrix in memory
            y_offset = offset2.detach().clone() # detach ensures that gradients do not propagate from y_offset to offset2; clone creates an independent matrix in memory
            
            if if_offset:
                z_offset = z_offset.permute(2, 0, 1, 3, 4, 5) # [K, N, C, D, W, H] permute to prepare for offset in kernel direction
                y_offset = y_offset.permute(2, 0, 1, 3, 4, 5) # [K, N, C, D, W, H]
                offset1 = offset1.permute(2, 0, 1, 3, 4, 5) # [K, N, C, D, W, H]
                offset2 = offset2.permute(2, 0, 1, 3, 4, 5) # [K, N, C, D, W, H]
                center = int(self.num_points // 2)
                z_offset[center] = 0
                y_offset[center] = 0
                for index in range(1, center + 1):
                    src_pos_z = offset1[center + index, :, :, :, :, index:]  # shape [N, C, D, W, H - index]
                    padded_pos_z = torch.nn.functional.pad(src_pos_z, (0, index, 0, 0, 0, 0), 'replicate')  # pad H dim at end
                    z_offset[center + index] = z_offset[center + index - 1] + padded_pos_z
                    
                    src_neg_z = offset1[center - index, :, :, :, :, :self.height-index]  # shape [N, C, D, W, H - index]
                    padded_neg_z = torch.nn.functional.pad(src_neg_z, (index, 0, 0, 0, 0, 0), 'replicate')  # pad H dim at start
                    z_offset[center - index] = z_offset[center - index + 1] + padded_neg_z

                    src_pos_y = offset2[center + index, :, :, :, :, index:]  # shape [N, C, D, W, H - index]
                    padded_pos_y = torch.nn.functional.pad(src_pos_y, (0, index, 0, 0, 0, 0), 'replicate')  # pad H dim at end
                    y_offset[center + index] = y_offset[center + index - 1] + padded_pos_y
                    
                    src_neg_y = offset2[center - index, :, :, :, :, :self.height-index]  # shape [N, C, D, W, H - index]
                    padded_neg_y = torch.nn.functional.pad(src_neg_y, (index, 0, 0, 0, 0, 0), 'replicate')  # pad H dim at start
                    y_offset[center - index] = y_offset[center - index + 1] + padded_neg_y

                z_offset = z_offset.permute(1, 2, 0, 3, 4, 5).to(self.device) # [N, C, K, D, W, H]
                y_offset = y_offset.permute(1, 2, 0, 3, 4, 5).to(self.device) # [N, C, K, D, W, H]
                z_new = z_new.add(z_offset) # add z_offset to z_new (which is all zeros except for depth)
                y_new = y_new.add(y_offset) # add y_offset to y_new (which is all zeros except for width)

            return z_new, y_new, x_new

        elif self.morph == 1:
            y_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
            y_spread = y_spread.view(1, 1, self.num_points, 1, 1, 1) # [1, 1, K, 1, 1, 1]
            y_grid = y_spread.expand(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height) # (N, C, K, D, W, H)

            z_new = z_center.to(self.device)
            y_new = (y_center + y_grid).to(self.device)
            x_new = x_center.to(self.device)                

            x_offset = offset1.detach().clone()
            z_offset = offset2.detach().clone() # [N, C, K, D, W, H]

            if if_offset:
                x_offset = x_offset.permute(2, 0, 1, 3, 4, 5) # [K, N, C, D, W, H]
                z_offset = z_offset.permute(2, 0, 1, 3, 4, 5)
                offset1 = offset1.permute(2, 0, 1, 3, 4, 5)
                offset2 = offset2.permute(2, 0, 1, 3, 4, 5)
                center = int(self.num_points // 2)
                x_offset[center] = 0
                z_offset[center] = 0
                for index in range(1, center + 1):
                    src_pos_x = offset1[center + index, :, :, :, index:, :]  # shape [N, C, D, W - index, H]
                    padded_pos_x = torch.nn.functional.pad(src_pos_x, (0, 0, 0, index, 0, 0), 'replicate')  # pad W dim at end
                    x_offset[center + index] = x_offset[center + index - 1] + padded_pos_x
                    
                    src_neg_x = offset1[center - index, :, :, :, :self.width-index, :]  # shape [N, C, D, W - index, H]
                    padded_neg_x = torch.nn.functional.pad(src_neg_x, (0, 0, index, 0, 0, 0), 'replicate')  # pad W dim at start
                    x_offset[center - index] = x_offset[center - index + 1] + padded_neg_x

                    src_pos_z = offset2[center + index, :, :, :, index:, :]  # shape [N, C, D, W - index, H]
                    padded_pos_z = torch.nn.functional.pad(src_pos_z, (0, 0, 0, index, 0, 0), 'replicate')  # pad W dim at end
                    z_offset[center + index] = z_offset[center + index - 1] + padded_pos_z
                    
                    src_neg_z = offset2[center - index, :, :, :, :self.width-index, :]  # shape [N, C, D, W - index, H]
                    padded_neg_z = torch.nn.functional.pad(src_neg_z, (0, 0, index, 0, 0, 0), 'replicate')  # pad W dim at start
                    z_offset[center - index] = z_offset[center - index + 1] + padded_neg_z

                x_offset = x_offset.permute(1, 2, 0, 3, 4, 5).to(self.device) # [N, C, K, D, W, H]
                z_offset = z_offset.permute(1, 2, 0, 3, 4, 5).to(self.device)
                z_new = z_new.add(z_offset)
                x_new = x_new.add(x_offset)

            return z_new, y_new, x_new

        else:
            z_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
            z_spread = z_spread.view(1, 1, self.num_points, 1, 1, 1) # [1, 1, K, 1, 1, 1]
            z_grid = z_spread.expand(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height) # (N, C, K, D, W, H)

            z_new = (z_center + z_grid).to(self.device)
            y_new = y_center.to(self.device)
            x_new = x_center.to(self.device)   

            # z = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
            # y = torch.linspace(0, 0, 1)
            # x = torch.linspace(0, 0, 1)
            # z, y, x = torch.meshgrid(z, y, x)
            # z_spread = z.reshape(-1, 1)
            # y_spread = y.reshape(-1, 1)
            # x_spread = x.reshape(-1, 1)

            # z_grid = z_spread.repeat([self.num_channels, self.depth * self.width * self.height])
            # z_grid = z_grid.reshape([self.num_channels, self.num_points, self.depth, self.width, self.height])
            # z_grid = z_grid.unsqueeze(0)  # [N, C, K, D, W, H]

            # y_grid = y_spread.repeat([self.num_channels, self.depth * self.width * self.height])
            # y_grid = y_grid.reshape([self.num_channels, self.num_points, self.depth, self.width, self.height])
            # y_grid = y_grid.unsqueeze(0)  # [N, C, K, D, W, H]

            # x_grid = x_spread.repeat([self.num_channels, self.depth * self.width * self.height])
            # x_grid = x_grid.reshape([self.num_channels, self.num_points, self.depth, self.width, self.height])
            # x_grid = x_grid.unsqueeze(0)  # [N, C, K, D, W, H]

            # z_new = z_center + z_grid
            # y_new = y_center + y_grid
            # x_new = x_center + x_grid  # [N, C, K, D, W, H]

            # z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1, 1) 
            # y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1, 1)
            # x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1, 1) # [N, C, K, D, W, H]
            
            # z_new = z_new.to(self.device)
            # y_new = y_new.to(self.device)
            # x_new = x_new.to(self.device)

            x_offset = offset1.detach().clone()
            y_offset = offset2.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(2, 0, 1, 3, 4, 5) # [K, N, C, D, W, H]
                y_offset = y_offset.permute(2, 0, 1, 3, 4, 5)
                offset1 = offset1.permute(2, 0, 1, 3, 4, 5)
                offset2 = offset2.permute(2, 0, 1, 3, 4, 5)
                center = int(self.num_points // 2)
                x_offset[center] = 0
                y_offset[center] = 0
                for index in range(1, center + 1):
                    src_pos_x = offset1[center + index, :, :, index:, :, :]  # shape [N, C, D-index, W, H]
                    padded_pos_x = torch.nn.functional.pad(src_pos_x, (0, 0, 0, 0, 0, index), 'replicate')  # pad D dim at end
                    x_offset[center + index] = x_offset[center + index - 1] + padded_pos_x
                    
                    src_neg_x = offset1[center - index, :, :, :self.depth-index, :, :]  # shape [N, C, D-index, W, H]
                    padded_neg_x = torch.nn.functional.pad(src_neg_x, (0, 0, 0, 0, index, 0), 'replicate')  # pad D dim at start
                    x_offset[center - index] = x_offset[center - index + 1] + padded_neg_x

                    src_pos_y = offset2[center + index, :, :, index:, :, :]  # shape [N, C, D-index, W, H]
                    padded_pos_y = torch.nn.functional.pad(src_pos_y, (0, 0, 0, 0, 0, index), 'replicate')  # pad D dim at end
                    y_offset[center + index] = y_offset[center + index - 1] + padded_pos_y
                    
                    src_neg_y = offset2[center - index, :, :, :self.depth-index, :, :]  # shape [N, C, D-index, W, H]
                    padded_neg_y = torch.nn.functional.pad(src_neg_y, (0, 0, 0, 0, index, 0), 'replicate')  # pad D dim at start
                    y_offset[center - index] = y_offset[center - index + 1] + padded_neg_y

                x_offset = x_offset.permute(1, 2, 0, 3, 4, 5).to(self.device) # [N, C, K, D, W, H]
                y_offset = y_offset.permute(1, 2, 0, 3, 4, 5).to(self.device)
                x_new = x_new.add(x_offset)
                y_new = y_new.add(y_offset)

            return z_new, y_new, x_new

    '''
    input: input feature map [N,C,D,W,H]；coordinate maps [N,C,K,D,W,H] 
    output: [N,C,D,W,K*H] or [N,C,D,K*W,H] or [N,C,K*D,W,H] deformed feature map
    '''
    def _vectorized_new_bilinear_interpolate_3D(self, input_feature, z, y, x):
        N, C, K, D, W, H = z.shape

        # Fold N and C into a single batch dimension, grid_sample treats C as independent batch items
        input_feature = input_feature.unsqueeze(2) # [N, C, 1, D, W, H]
        input_feature = input_feature.expand(N, C, K, D, W, H) # [N, C, K, D, W, H]
        input_feature = input_feature.reshape(N*C, 1, K*D, W, H).float() # [N*C, 1, K*D, W, H]

        # Normalise coordinates to [-1, 1]
        z_norm = 2.0 * z / (D - 1) - 1.0
        z_norm = z_norm.clamp(min=-1.0, max=1.0)
        y_norm = 2.0 * y / (W - 1) - 1.0
        y_norm = y_norm.clamp(min=-1.0, max=1.0)
        x_norm = 2.0 * x / (H - 1) - 1.0
        x_norm = x_norm.clamp(min=-1.0, max=1.0)
        
        # Build grid and fold N, C into batch dimension
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1) # [N, C, K, D, W, H, 3]
        grid = grid.reshape(N*C, K*D, W, H, 3).float() # [N*C, K*D, W, H, 3]

        outputs = torch.nn.functional.grid_sample(
            input_feature, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # [N*C, 1, K*D, W, H]

        outputs = outputs.reshape(N, C, K, D, W, H)

        if self.morph == 0:
            outputs = outputs.permute(0, 1, 3, 4, 2, 5)
            outputs = outputs.reshape(N, C, D, W, K*H)
        elif self.morph == 1:
            outputs = outputs.permute(0, 1, 3, 2, 4, 5)
            outputs = outputs.reshape(N, C, D, K*W, H)
        else:
            outputs = outputs.reshape(N, C, K*D, W, H)

        return outputs


    '''
    input: input feature map [N,C,D,W,H]；coordinate maps [N,C,K,D,W,H] 
    ooutput: [N,C,D,W,K*H] or [N,C,D,K*W,H] or [N,C,K*D,W,H] deformed feature map
    '''
    def _new_bilinear_interpolate_3D(self, input_feature, z, y, x):
        num_batch = input_feature.shape[0] # variable num_batch
        outputs = torch.zeros(num_batch, self.num_channels, self.num_points, self.depth, self.width, self.height).to(self.device)
        
        z_norm = 2.0 * z / (self.depth - 1) - 1.0
        z_norm = z_norm.clamp(min=-1.0, max=1.0)
        y_norm = 2.0 * y / (self.width - 1) - 1.0
        y_norm = y_norm.clamp(min=-1.0, max=1.0)
        x_norm = 2.0 * x / (self.height - 1) - 1.0
        x_norm = x_norm.clamp(min=-1.0, max=1.0)
        
        for i in range(0, num_batch):
            for j in range(0, self.num_channels):
                input_feature_slice = input_feature[i][j]
                input_feature_slice = input_feature_slice.unsqueeze(0)
                input_feature_slice = input_feature_slice.unsqueeze(0)
                input_feature_slice = input_feature_slice.expand(self.num_points, 1, self.depth, self.width, self.height).float().to(self.device)
                z_slice = z_norm[i][j]
                y_slice = y_norm[i][j]
                x_slice = x_norm[i][j]
                offset_slice = torch.zeros(3, self.num_points, self.depth, self.width, self.height)
                offset_slice[0] = z_slice
                offset_slice[1] = y_slice
                offset_slice[2] = x_slice
                offset_slice = offset_slice.permute(1,2,3,4,0).float().to(self.device)
                outputs[i][j] = torch.nn.functional.grid_sample(input_feature_slice, offset_slice, mode="bilinear", padding_mode="border", align_corners=True).squeeze()
        
        if self.morph == 0:
            outputs = outputs.permute(0, 1, 3, 4, 2, 5)
            outputs = outputs.reshape([num_batch, self.num_channels, self.depth, self.width, self.num_points*self.height])
        elif self.morph == 1:
            outputs = outputs.permute(0, 1, 3, 2, 4, 5)
            outputs = outputs.reshape([num_batch, self.num_channels, self.depth, self.num_points*self.width, self.height])
        else:
            outputs = outputs.reshape([num_batch, self.num_channels, self.num_points*self.depth, self.width, self.height])
        return outputs

    '''
    input: input feature map [N,C,D,W,H]；coordinate map [N,C,K,D,W,H] 
    output: [N,C,D,W,K*H] or [N,C,D,K*W,H] or [N,C,K*D,W,H] deformed feature map
    '''
    def _bilinear_interpolate_3D(self, input_feature, z, y, x):
        num_batch = input_feature.shape[0] # variable num_batch
        z = z.reshape([-1]).float() # [N*C*K*D*W*H]
        y = y.reshape([-1]).float() # [N*C*K*D*W*H]
        x = x.reshape([-1]).float() # [N*C*K*D*W*H]

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
        x1 = torch.clamp(x1, zero, max_x)  # [N*C*K*D*W*H]

        # convert input_feature and coordinate X, Y to 3D，for gathering
        # input_feature_flat = input_feature.reshape([-1, self.num_channels])   # [N*D*W*H, C]
        input_feature_flat = input_feature.flatten() # [N*C*D*W*H]
        input_feature_flat = input_feature_flat.reshape(num_batch, self.num_channels, self.depth, self.width,
                                                        self.height) # [N, C, D, W, H] 
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 4, 1) # [N, D, W, H, C] 
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels) # [N*D*W*H, C]
        dimension = self.num_channels * self.height * self.width * self.depth
        channel_dimension = self.height * self.width * self.depth

        base = torch.arange(num_batch) * dimension # [N] 
        base = base.reshape([-1, 1]).float()  # [N,1]

        base_repeat = torch.ones(self.num_channels).unsqueeze(0).float() # [1, C]
        base = torch.matmul(base, base_repeat) # [N, C] all batch_num*1
        base = base.reshape(-1) # [N*C] all batch_num*1

        channel_base = torch.arange(self.num_channels) * channel_dimension # [C]
        channel_base = channel_base.unsqueeze(0) # [1, C]
        channel_base = channel_base.repeat(num_batch, 1) # [N, C]
        channel_base = channel_base.reshape(-1).float() # [N*C]
        
        base = (base + channel_base).unsqueeze(1) # [N*C, 1]

        repeat = torch.ones([self.num_points * self.depth * self.width * self.height]).unsqueeze(0) # unsqueeze returns a new tensor with a dimension of 1 inserted in the indicated location; [1, K*D*W*H]
        repeat = repeat.float()  # [1, K*D*W*H]

        base = torch.matmul(base, repeat)  # [N*C, 1] * [1, K*D*W*H]  ==> [N*C, K*D*W*H] all channel_num*dimension
        base = base.reshape([-1]) # [N*C*K*D*W*H] 

        base = base.to(self.device) # what is the point of this? torch.arange and torch.repeat are done on CPU by default. Have to move to same device as the other tensors.

        base_z0 = base + z0 * self.height * self.width
        base_z1 = base + z1 * self.height * self.width
        base_y0 = base + y0 * self.height # why no width? because the reshaping flattened everything into one axis. To properly get the index, you have to multiply differently.
        base_y1 = base + y1 * self.height 

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 + base_z0 - base + x0
        index_b0 = base_y0 + base_z1 - base + x0
        index_c0 = base_y0 + base_z0 - base + x1
        index_d0 = base_y0 + base_z1 - base + x1  # [N*C*K*D*W*H] 

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 + base_z0 - base + x0
        index_b1 = base_y1 + base_z1 - base + x0
        index_c1 = base_y1 + base_z0 - base + x1
        index_d1 = base_y1 + base_z1 - base + x1  # [N*C*K*D*W*H] 

        # get 8 grid values  ([N*D*W*H,C], [N*D*W*H*27])
        value_a0 = input_feature_flat[index_a0.type(torch.int64)]
        value_b0 = input_feature_flat[index_b0.type(torch.int64)]
        value_c0 = input_feature_flat[index_c0.type(torch.int64)]
        value_d0 = input_feature_flat[index_d0.type(torch.int64)]
        value_a1 = input_feature_flat[index_a1.type(torch.int64)]
        value_b1 = input_feature_flat[index_b1.type(torch.int64)]
        value_c1 = input_feature_flat[index_c1.type(torch.int64)]
        value_d1 = input_feature_flat[index_d1.type(torch.int64)]  # [N*C*K*D*W*H] 

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
        x1 = torch.clamp(x1, zero, max_x + 1)  

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        z0_float = z0.float()
        z1_float = z1.float()

        vol_a0 = ((z1_float - z) * (y1_float - y) * (x1_float - x))  
        vol_b0 = ((z - z0_float) * (y1_float - y) * (x1_float - x))
        vol_c0 = ((z1_float - z) * (y1_float - y) * (x - x0_float))
        vol_d0 = ((z - z0_float) * (y1_float - y) * (x - x0_float))
        vol_a1 = ((z1_float - z) * (y - y0_float) * (x1_float - x))
        vol_b1 = ((z - z0_float) * (y - y0_float) * (x1_float - x))
        vol_c1 = ((z1_float - z) * (y - y0_float) * (x - x0_float))
        vol_d1 = ((z - z0_float) * (y - y0_float) * (x - x0_float))

        outputs = value_a0 * vol_a0 + value_b0 * vol_b0 + value_c0 * vol_c0 + value_d0 * vol_d0 + value_a1 * vol_a1 + value_b1 * vol_b1 + value_c1 * vol_c1 + value_d1 * vol_d1

        if self.morph == 0:
            outputs = outputs.reshape([num_batch, self.depth, self.width, self.num_points*self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        elif self.morph == 1:
            outputs = outputs.reshape([num_batch, self.depth, self.num_points*self.width, self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        else:
            outputs = outputs.reshape([num_batch, self.num_points*self.depth, self.width, self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        z, y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._vectorized_new_bilinear_interpolate_3D(input, z, y, x)
        return deformed_feature
