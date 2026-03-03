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
        self.regular_conv = nn.Conv3d(in_ch, 3 * self.kernel_size, 3, padding=1) # change this for testing
        self.bn = nn.BatchNorm3d(3 * self.kernel_size) # Normalizes across the Channel dimension; returns same shape as input
        self.device = device

        #self.if_offset = if_offset
        #self.morph = morph
        #self.extend_scope = extend_scope

        #self.dcn_conv_x = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, self.kernel_size), stride=(1, 1, self.kernel_size), padding=0)  # conv in x-direction
        #self.dcn_conv_y = nn.Conv3d(in_ch, out_ch, kernel_size=(1, self.kernel_size, 1), stride=(1, self.kernel_size, 1), padding=0)  # conv in y-direction
        #self.dcn_conv_z = nn.Conv3d(in_ch, out_ch, kernel_size=(self.kernel_size, 1, 1), stride=(self.kernel_size, 1, 1), padding=0)  # conv in z-direction

        self.dcn_conv = nn.Conv3d(3 * self.kernel_size, out_ch, kernel_size=1, stride=1, padding=0)

        #self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f):
        # Input: [N, K, D, W, H];
        output = self.regular_conv(f) # Output: [N, 3*K, D, W, H];
        output = self.bn(output) # Output: [N, 3*K, D, W, H];
        output = torch.tanh(output) # Output: [N, 3*K, D, W, H]; tanh is (-1, 1)
        #input_shape = f.shape # shape: [N, C, D, W, H];

        #dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)
        #deformed_feature = dcn.deform_conv(f, output, self.if_offset) # _coordinate_map_3D (Output: [N, D, W, H*K] OR [N, D, W*K, H] OR [N, D*K, W, H]) + _bilinear_interpolate_3D (Output: [N, 1, D, W, H*K] etc.)

        # Only ever does one of the following
        #if self.morph == 0:
            #x = self.dcn_conv_x(deformed_feature)
            #x = self.gn(x)
            #x = self.relu(x)
            #return x
        #elif self.morph == 1:
            #x = self.dcn_conv_y(deformed_feature)
            #x = self.gn(x)
            #x = self.relu(x)
            #return x
        #else:
            #x = self.dcn_conv_z(deformed_feature)
            #x = self.gn(x)
            #x = self.relu(x)  
            #return x
        
        x = self.dcn_conv(output)
        x = self.gn(x)
        x = self.relu(x)
        return x
