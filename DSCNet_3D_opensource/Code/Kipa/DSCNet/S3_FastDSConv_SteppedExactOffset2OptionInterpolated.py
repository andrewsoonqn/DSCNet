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
        self.offset_conv = nn.Conv3d(in_ch, 2 * 2 * self.kernel_size, 3, padding=1) # takes in [N, C, D, W, H] and returns [N, 2*2*K, D, W, H]
        self.bn = nn.BatchNorm3d(2 * 2 * self.kernel_size) # Normalizes across the channel dimension;
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
        self.dropout = nn.Dropout(p=0.25, inplace=False)

        self.dcn = None


    def forward(self, f):
        # Input: [N, K, D, W, H];
        offset = self.offset_conv(f) # Output: [N, 2*2*K, D, W, H];
        offset = self.bn(offset) # Output: [N, 2*2*K, D, W, H];
        offset = torch.tanh(offset) # Output: [N, 2*2*K, D, W, H]; tanh is (-1, 1)
        input_shape = f.shape # shape: [N, C, D, W, H];

        if self.dcn is None:
            self.dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)

        deformed_feature = self.dcn.deform_conv(f, offset, self.if_offset) # _coordinate_map_3D (Output: [N, K, D, W, H]) + _vectorized_new_bilinear_interpolate_3D (Output: [N, C, D, W, H*K] OR [N, C, D, W*K, H] OR [N, C, D*K, W, H])

        # Only ever does one of the following
        if self.morph == 0:
            x = self.dcn_conv_x(deformed_feature) # [N, C_out, D, W, H]
            x = self.gn(x)
            x = self.relu(x)
            #x = self.dropout(x)
            return x
        elif self.morph == 1:
            x = self.dcn_conv_y(deformed_feature) # [N, C_out, D, W, H]
            x = self.gn(x)
            x = self.relu(x)
            #x = self.dropout(x)
            return x
        else:
            x = self.dcn_conv_z(deformed_feature) # [N, C_out, D, W, H]
            x = self.gn(x)
            x = self.relu(x)  
            #x = self.dropout(x)
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
    input: offset [N,2*2*K,D,W,H]
    output: [N,K,D,W,H]   coordinate map
    '''
    def _coordinate_map_3D(self, offset, if_offset):
        # variable num_batch
        num_batch = offset.shape[0] # N

        # offset
        offset1, offset2 = torch.split(offset, 2*self.num_points, dim=1) # Split offset into groups of self.num_points i.e. [N, 2*K, D, W, H]
        
        z_coords = torch.arange(self.depth) # [D]
        y_coords = torch.arange(self.width) # [W]
        x_coords = torch.arange(self.height) # [H]
        z_center, y_center, x_center = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij') # [D, W, H] where z increases in depth direction, y in width, and x in height.
        z_center = z_center.view(1, 1, self.depth, self.width, self.height) # [1, 1, D, W, H]
        y_center = y_center.view(1, 1, self.depth, self.width, self.height) # [1, 1, D, W, H]
        x_center = x_center.view(1, 1, self.depth, self.width, self.height) # [1, 1, D, W, H]
        z_center = z_center.expand(num_batch, self.num_points, self.depth, self.width, self.height).float() # [N, K, D, W, H]
        y_center = y_center.expand(num_batch, self.num_points, self.depth, self.width, self.height).float() # [N, K, D, W, H]
        x_center = x_center.expand(num_batch, self.num_points, self.depth, self.width, self.height).float() # [N, K, D, W, H]

        if self.morph == 0:
            x_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
            x_spread = x_spread.view(1, self.num_points, 1, 1, 1) # [1, K, 1, 1, 1]
            x_grid = x_spread.expand(num_batch, self.num_points, self.depth, self.width, self.height) # (N, K, D, W, H)

            z_new = z_center.detach().clone().to(self.device) # (N, K, D, W, H)
            y_new = y_center.detach().clone().to(self.device) # (N, K, D, W, H)
            x_new = (x_center + x_grid).to(self.device) # (N, K, D, W, H)

            if if_offset: # x_new are positions, offsetX are vectors in (-1, 1)
                z_new = z_new.permute(1, 0, 2, 3, 4) # [K, N, D, W, H] permute to prepare for offset in kernel direction
                y_new = y_new.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
                x_new = x_new.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
                offset1 = offset1.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
                offset2 = offset2.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
                center = int(self.num_points // 2)

                z_new[center + 1] = z_new[center] + 0.5*offset1[(center + 1) * 2] + 0.5*offset1[(center + 1) * 2 + 1]
                y_new[center + 1] = y_new[center] + 0.5*offset2[(center + 1) * 2] + 0.5*offset2[(center + 1) * 2 + 1]
                z_new[center - 1] = z_new[center] + 0.5*offset1[(center - 1) * 2] + 0.5*offset1[(center - 1) * 2 + 1]
                y_new[center - 1] = y_new[center] + 0.5*offset2[(center - 1) * 2] + 0.5*offset2[(center - 1) * 2 + 1] # for center to center
                z_new = z_new.clamp(min=0, max=self.depth)
                y_new = y_new.clamp(min=0, max=self.width)

                for index in range(2, center + 1):
                    offset_cat_pos = torch.cat((offset1[(center + index) * 2 : (center + index) * 2 + 2], offset2[(center + index) * 2 : (center + index) * 2 + 2]), dim=0) # [4, N, D, W, H]
                    z_offset_pos, y_offset_pos = self._offset_interpolate_3D(offset_cat_pos, z_new[center + index - 1], y_new[center + index - 1], x_new[center + index - 1]) # [2, N, D, W, H]
                    z_index_pos = (z_new[center + index - 1] - z_new[center + index - 2] + 1) / 2 # [N, D, W, H]
                    y_index_pos = (y_new[center + index - 1] - y_new[center + index - 2] + 1) / 2 # [N, D, W, H]
                    z_new[center + index] = (z_new[center + index - 1] + z_index_pos * z_offset_pos[0] + (1 - z_index_pos) * z_offset_pos[1]).clamp(min=0, max=self.depth) # [N, D, W, H]
                    y_new[center + index] = (y_new[center + index - 1] + y_index_pos * y_offset_pos[0] + (1 - y_index_pos) * y_offset_pos[1]).clamp(min=0, max=self.width) # [N, D, W, H]
                    
                    offset_cat_neg = torch.cat((offset1[(center - index) * 2 : (center - index) * 2 + 2], offset2[(center - index) * 2 : (center - index) * 2 + 2]), dim=0) # [4, N, D, W, H]
                    z_offset_neg, y_offset_neg = self._offset_interpolate_3D(offset_cat_neg, z_new[center - index + 1], y_new[center - index + 1], x_new[center - index + 1]) # [2, N, D, W, H]
                    z_index_neg = (z_new[center - index + 1] - z_new[center - index + 2] + 1) / 2 # [N, D, W, H]
                    y_index_neg = (y_new[center - index + 1] - y_new[center - index + 2] + 1) / 2 # [N, D, W, H]
                    z_new[center - index] = (z_new[center - index + 1] + z_index_neg * z_offset_neg[0] + (1 - z_index_neg) * z_offset_neg[1]).clamp(min=0, max=self.depth) # [N, D, W, H]
                    y_new[center - index] = (y_new[center - index + 1] + y_index_neg * y_offset_neg[0] + (1 - y_index_neg) * y_offset_neg[1]).clamp(min=0, max=self.width) # [N, D, W, H]
                
                z_new = z_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]
                y_new = y_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]
                x_new = x_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]

            return z_new, y_new, x_new

        elif self.morph == 1:
            y_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
            y_spread = y_spread.view(1, self.num_points, 1, 1, 1) # [1, K, 1, 1, 1]
            y_grid = y_spread.expand(num_batch, self.num_points, self.depth, self.width, self.height) # (N, K, D, W, H)

            x_new = x_center.detach().clone().to(self.device)
            z_new = z_center.detach().clone().to(self.device)
            y_new = (y_center + y_grid).to(self.device)  

            if if_offset:
                x_new = x_new.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
                z_new = z_new.permute(1, 0, 2, 3, 4)
                y_new = y_new.permute(1, 0, 2, 3, 4)
                offset1 = offset1.permute(1, 0, 2, 3, 4)
                offset2 = offset2.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)

                x_new[center + 1] = x_new[center] + 0.5*offset1[(center + 1) * 2] + 0.5*offset1[(center + 1) * 2 + 1]
                z_new[center + 1] = z_new[center] + 0.5*offset2[(center + 1) * 2] + 0.5*offset2[(center + 1) * 2 + 1]
                x_new[center - 1] = x_new[center] + 0.5*offset1[(center - 1) * 2] + 0.5*offset1[(center - 1) * 2 + 1]
                z_new[center - 1] = z_new[center] + 0.5*offset2[(center - 1) * 2] + 0.5*offset2[(center - 1) * 2 + 1]
                x_new = x_new.clamp(min=0, max=self.height)
                z_new = z_new.clamp(min=0, max=self.depth)

                for index in range(2, center + 1):
                    offset_cat_pos = torch.cat((offset1[(center + index) * 2 : (center + index) * 2 + 2], offset2[(center + index) * 2 : (center + index) * 2 + 2]), dim=0)
                    x_offset_pos, z_offset_pos = self._offset_interpolate_3D(offset_cat_pos, z_new[center + index - 1], y_new[center + index - 1], x_new[center + index - 1])
                    x_index_pos = (x_new[center + index - 1] - x_new[center + index - 2] + 1) / 2
                    z_index_pos = (z_new[center + index - 1] - z_new[center + index - 2] + 1) / 2
                    x_new[center + index] = (x_new[center + index - 1] + x_index_pos * x_offset_pos[0] + (1 - x_index_pos) * x_offset_pos[1]).clamp(min=0, max=self.height)
                    z_new[center + index] = (z_new[center + index - 1] + z_index_pos * z_offset_pos[0] + (1 - z_index_pos) * z_offset_pos[1]).clamp(min=0, max=self.depth)
                    
                    offset_cat_neg = torch.cat((offset1[(center - index) * 2 : (center - index) * 2 + 2], offset2[(center - index) * 2 : (center - index) * 2 + 2]), dim=0)
                    x_offset_neg, z_offset_neg = self._offset_interpolate_3D(offset_cat_neg, z_new[center - index + 1], y_new[center - index + 1], x_new[center - index + 1])
                    x_index_neg = (x_new[center - index + 1] - x_new[center - index + 2] + 1) / 2
                    z_index_neg = (z_new[center - index + 1] - z_new[center - index + 2] + 1) / 2
                    x_new[center - index] = (x_new[center - index + 1] + x_index_neg * x_offset_neg[0] + (1 - x_index_neg) * x_offset_neg[1]).clamp(min=0, max=self.height)
                    z_new[center - index] = (z_new[center - index + 1] + z_index_neg * z_offset_neg[0] + (1 - z_index_neg) * z_offset_neg[1]).clamp(min=0, max=self.depth)
                
                x_new = x_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]
                z_new = z_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]
                y_new = y_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]

            return z_new, y_new, x_new

        else:
            z_spread = torch.linspace(-self.num_points // 2, self.num_points // 2, self.num_points) # [K] running from -self.num_points//2 to self.num_points//2
            z_spread = z_spread.view(1, self.num_points, 1, 1, 1) # [1, K, 1, 1, 1]
            z_grid = z_spread.expand(num_batch, self.num_points, self.depth, self.width, self.height) # (N, K, D, W, H)
            
            x_new = x_center.detach().clone().to(self.device)
            y_new = y_center.detach().clone().to(self.device)
            z_new = (z_center + z_grid).to(self.device)

            if if_offset:
                x_new = x_new.permute(1, 0, 2, 3, 4) # [K, N, D, W, H]
                y_new = y_new.permute(1, 0, 2, 3, 4)
                z_new = z_new.permute(1, 0, 2, 3, 4)
                offset1 = offset1.permute(1, 0, 2, 3, 4)
                offset2 = offset2.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)

                x_new[center + 1] = x_new[center] + 0.5*offset1[(center + 1) * 2] + 0.5*offset1[(center + 1) * 2 + 1]
                y_new[center + 1] = y_new[center] + 0.5*offset2[(center + 1) * 2] + 0.5*offset2[(center + 1) * 2 + 1]
                x_new[center - 1] = x_new[center] + 0.5*offset1[(center - 1) * 2] + 0.5*offset1[(center - 1) * 2 + 1]
                y_new[center - 1] = y_new[center] + 0.5*offset2[(center - 1) * 2] + 0.5*offset2[(center - 1) * 2 + 1]
                x_new = x_new.clamp(min=0, max=self.height)
                y_new = y_new.clamp(min=0, max=self.width)
                
                for index in range(2, center + 1):
                    offset_cat_pos = torch.cat((offset1[(center + index) * 2 : (center + index) * 2 + 2], offset2[(center + index) * 2 : (center + index) * 2 + 2]), dim=0)
                    x_offset_pos, y_offset_pos = self._offset_interpolate_3D(offset_cat_pos, z_new[center + index - 1], y_new[center + index - 1], x_new[center + index - 1])
                    x_index_pos = (x_new[center + index - 1] - x_new[center + index - 2] + 1) / 2
                    y_index_pos = (y_new[center + index - 1] - y_new[center + index - 2] + 1) / 2
                    x_new[center + index] = (x_new[center + index - 1] + x_index_pos * x_offset_pos[0] + (1 - x_index_pos) * x_offset_pos[1]).clamp(min=0, max=self.height)
                    y_new[center + index] = (y_new[center + index - 1] + y_index_pos * y_offset_pos[0] + (1 - y_index_pos) * y_offset_pos[1]).clamp(min=0, max=self.width)
                    
                    offset_cat_neg = torch.cat((offset1[(center - index) * 2 : (center - index) * 2 + 2], offset2[(center - index) * 2 : (center - index) * 2 + 2]), dim=0)
                    x_offset_neg, y_offset_neg = self._offset_interpolate_3D(offset_cat_neg, z_new[center - index + 1], y_new[center - index + 1], x_new[center - index + 1])
                    x_index_neg = (x_new[center - index + 1] - x_new[center - index + 2] + 1) / 2
                    y_index_neg = (y_new[center - index + 1] - y_new[center - index + 2] + 1) / 2
                    x_new[center - index] = (x_new[center - index + 1] + x_index_neg * x_offset_neg[0] + (1 - x_index_neg) * x_offset_neg[1]).clamp(min=0, max=self.height)
                    y_new[center - index] = (y_new[center - index + 1] + y_index_neg * y_offset_neg[0] + (1 - y_index_neg) * y_offset_neg[1]).clamp(min=0, max=self.width)

                x_new = x_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]
                y_new = y_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]
                z_new = z_new.permute(1, 0, 2, 3, 4) # [N, K, D, W, H]

            return z_new, y_new, x_new

    '''
    input: input feature map [N,C,D,W,H]；coordinate maps [N,K,D,W,H] 
    output: [N,C,D,W,K*H] or [N,C,D,K*W,H] or [N,C,K*D,W,H] deformed feature map
    '''
    def _vectorized_new_bilinear_interpolate_3D(self, input_feature, z, y, x):
        N, K, D, W, H = z.shape
        C = self.num_channels

        # Fold K into D dimension
        input_feature = input_feature.unsqueeze(2) # [N, C, 1, D, W, H]
        input_feature = input_feature.expand(N, C, K, D, W, H) # [N, C, K, D, W, H]
        input_feature = input_feature.reshape(N, C, K*D, W, H).float() # [N, C, K*D, W, H]

        # Normalise coordinates to [-1, 1]
        z_norm = 2.0 * z / (D - 1) - 1.0
        z_norm = z_norm.clamp(min=-1.0, max=1.0)
        y_norm = 2.0 * y / (W - 1) - 1.0
        y_norm = y_norm.clamp(min=-1.0, max=1.0)
        x_norm = 2.0 * x / (H - 1) - 1.0
        x_norm = x_norm.clamp(min=-1.0, max=1.0)
        
        # Build grid
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1) # [N, K, D, W, H, 3]
        grid = grid.reshape(N, K*D, W, H, 3).float() # [N, K*D, W, H, 3]

        outputs = torch.nn.functional.grid_sample(
            input_feature, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # [N, C, K*D, W, H]

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
    input: offset map [4,N,D,W,H]；coordinate maps [N,D,W,H]
    output: interpolated offset map [2,N,D,W,H] 
    '''
    def _offset_interpolate_3D(self, offset_map, z, y, x):
        N, D, W, H = z.shape

        # Prepare offset_map shape for grid_sample
        offset_map = offset_map.permute(1, 0, 2, 3, 4).float() # [N, 4, D, W, H]

        # Normalise coordinates to [-1, 1]
        z_norm = 2.0 * z / (D - 1) - 1.0
        z_norm = z_norm.clamp(min=-1.0, max=1.0)
        y_norm = 2.0 * y / (W - 1) - 1.0
        y_norm = y_norm.clamp(min=-1.0, max=1.0)
        x_norm = 2.0 * x / (H - 1) - 1.0
        x_norm = x_norm.clamp(min=-1.0, max=1.0)
        
        # Build grid
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).float() # [N, D, W, H, 3]

        output = torch.nn.functional.grid_sample(
            offset_map, grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        ) # [N, 4, D, W, H]
        output = output.permute(1, 0, 2, 3, 4) # [4, N, D, W, H]

        return output[0:2], output[2:4]
        #return output
    
    def deform_conv(self, input, offset, if_offset):
        z, y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._vectorized_new_bilinear_interpolate_3D(input, z, y, x)
        return deformed_feature
