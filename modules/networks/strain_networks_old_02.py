# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:25:06 2021

@author: Jerry Xing
"""

import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
from utils.data import get_data_type_by_category


# from modules.networks.Siamese import NetSiamese, NetSimpleFCN
# def get_network_by_name(name, config = {}):
#     # if name =='NetStrainMatSectionClassify':        
#     #     net = NetStrainMatSectionClassify(config)
#     # elif name =='NetStrainMatJointClsPred':
#     #     net = NetStrainMatJointClsPred(config)
#     # elif name =='NetStrainMat2TOS':
#     #     net = NetStrainMat2TOS(config)
#     if name =='NetStrainMat2TOS':
#         net = NetStrainMat2TOS(config)
#     elif name == 'NetStrainMat2ClsReg':
#         net =  NetStrainMat2ClsReg(config)
#     elif name == 'NetStrainMat2ClsDistMapReg':
#         net =  NetStrainMat2ClsDistMapReg(config)
#     else:
#         raise ValueError('Unsupported net name: ', name)
#     return net

class NetStrainMat2TOS(nn.Module):
    def __init__(self, config={}):
        super(NetStrainMat2TOS, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors_in = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out = self.paras.get('n_sectors_out', 18)
        self.n_frames = self.paras.get('n_frames', 25)
        self.n_conv_layers = self.paras.get('n_conv_layers', 4)
        self.n_conv_channels = self.paras.get('n_conv_channels', 16)
        self.conv_size = self.paras.get('conv_size', 3)
        self.conv_stride = self.paras.get('conv_stride', 1)

        useBN = False
        # actiFunc = nn.Sigmoid()
        actiFunc = nn.ReLU(True)
        # actiFunc = nn.LeakyReLU(True)
        bnFuncConv1 = nn.BatchNorm2d(self.n_conv_channels) if useBN else nn.Identity()
        bnFuncConv2 = nn.BatchNorm2d(1) if useBN else nn.Identity()
        convs = [nn.Conv2d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=self.conv_stride,
                           padding=self.conv_size // 2), actiFunc, bnFuncConv1]
        for innerLayerIdx in range(self.n_conv_layers - 2):
            convs += [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=self.conv_stride,
                                padding=self.conv_size // 2), actiFunc, bnFuncConv1]
        convs += [
            nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=self.conv_stride, padding=self.conv_size // 2),
            actiFunc, bnFuncConv2]

        # linear = [nn.Flatten(), nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_out)]

        # linear = [nn.Flatten(), \
        #           nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_in*self.n_frames//2),\
        #           bnFunc, actiFunc, \
        #           nn.Linear(self.n_sectors_in*self.n_frames//2, self.n_sectors_out)]
        bnFuncLinear1 = nn.BatchNorm1d(self.n_sectors_in * self.n_frames // 2) if useBN else nn.Identity()
        bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in * self.n_frames // 4) if useBN else nn.Identity()
        linear = [nn.Flatten(), \
                  nn.Linear(self.n_sectors_in * self.n_frames, self.n_sectors_in * self.n_frames // 2), \
                  actiFunc, bnFuncLinear1, \
                  nn.Linear(self.n_sectors_in * self.n_frames // 2, self.n_sectors_in * self.n_frames // 4),
                  actiFunc, bnFuncLinear2, \
                  nn.Linear(self.n_sectors_in * self.n_frames // 4, self.n_sectors_out)]
        self.layers = nn.ModuleList(convs + linear)

    @autocast()
    def forward(self, input_dict):
        # with autocast():
        x = input_dict[self.input_type]  # .cuda()
        # print(self.input_type, x.dtype)
        # x = self.net(x)        
        for layer in self.layers:
            # print(x.shape)
            # print(layer)
            # print(layer.weight.type())
            x = layer(x)
        x = nn.LeakyReLU(inplace=True)(x - 17) + 17
        # return x
        return {self.output_types: x}

    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.input_type = input_types
        elif type(input_types) is list:
            self.input_type = get_data_type_by_category('strainmat', input_types)

    def set_output_types(self, output_types):
        if type(output_types) is str:
            self.output_types = output_types
        elif type(output_types) is list:
            self.output_types = get_data_type_by_category('TOS', output_types)


def conv_net(n_input_channels, n_conv_layers, n_conv_channels, conv_size,
             stride_size, padding_size, n_output_channels,
             n_pooling, actiFunc, useBN=False):
    # useBN = self.use_batch_norm
    # actiFunc = nn.Sigmoid()
    # actiFunc = nn.ReLU(True)
    # actiFunc = nn.LeakyReLU(True)
    if n_conv_layers > 1:
        convs_layers = [nn.Conv2d(n_input_channels, n_conv_channels, conv_size,
                                  stride=stride_size, padding=padding_size), actiFunc]
        if useBN:
            convs_layers += [nn.BatchNorm2d(n_conv_channels)]
    
        if n_pooling > 1:
            convs_layers += [nn.MaxPool2d(2, stride=2)]
            # joint_convs += [nn.MaxPool2d((1,2), stride = (1,2))]
    
        for innerLayerIdx in range(n_conv_layers - 2):
            convs_layers += [nn.Conv2d(n_conv_channels, n_conv_channels, conv_size,
                                       stride=stride_size, padding=padding_size),
                             actiFunc]
            if useBN:
                convs_layers += [nn.BatchNorm2d(n_conv_channels)]
        convs_layers += [nn.Conv2d(n_conv_channels, n_output_channels, conv_size,
                                   stride=stride_size, padding=padding_size), actiFunc]
        if useBN:
            convs_layers += [nn.BatchNorm2d(n_output_channels)]
    
        if n_pooling > 0:
            convs_layers += [nn.MaxPool2d(2, stride=2)]
    else:
        convs_layers = [nn.Conv2d(n_input_channels, n_output_channels, conv_size,
                                  stride=stride_size, padding=padding_size), actiFunc]
        if useBN:
            convs_layers += [nn.BatchNorm2d(n_output_channels)]

    return convs_layers
    # return nn.ModuleList(joint_convs)


# def fcn_OLD(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc,
#         useBN=False):
#     if n_linear_layers > 1:
#         linear_layers = [nn.Flatten(),
#                          nn.Linear(linear_input_feature_dim, linear_inner_feature_dim),
#                          actiFunc]
#         if useBN:
#             linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
        
#         if linear_inner_feature_dim is None:
#             linear_inner_feature_dim = np.linspace(linear_inner_feature_dim, linear_output_feature_dim, n_linear_layers)[1:-1]
#         elif type(linear_inner_feature_dim) is int:
#             linear_inner_feature_dim = [linear_inner_feature_dim] * (n_linear_layers - 2)
        
#         for inner_layer_idx in range(n_linear_layers - 2):
#             curr_layer_dim = linear_inner_feature_dim[inner_layer_idx]
#             # linear_layers += [nn.Linear(linear_inner_feature_dim, linear_inner_feature_dim), actiFunc]
#             linear_layers += [nn.Linear(linear_inner_feature_dim, linear_inner_feature_dim), actiFunc]
#             if useBN:
#                 # linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
    
#         linear_layers += [nn.Linear(linear_inner_feature_dim, linear_output_feature_dim)]
#     else:
#         linear_layers = [nn.Flatten(),
#                          nn.Linear(linear_input_feature_dim, linear_output_feature_dim),
#                          actiFunc]
#         if useBN:
#             linear_layers += [nn.BatchNorm1d(linear_output_feature_dim)]
#     return linear_layers

def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc,
        useBN=False):
    
    if n_linear_layers > 1:
        layers = [nn.Flatten()]
        
        
        if linear_inner_feature_dim is None:
            layer_input_dims = np.linspace(linear_input_feature_dim, linear_output_feature_dim, n_linear_layers + 1).astype(int)
        elif type(linear_inner_feature_dim) is int:
            layer_input_dims = [linear_input_feature_dim] + [linear_inner_feature_dim]*(n_linear_layers-2+1) + [linear_output_feature_dim]
                                
        for layer_idx in range(n_linear_layers):
            curr_layer_input_dim = layer_input_dims[layer_idx]
            curr_layer_output_dim = layer_input_dims[layer_idx + 1]
                        
            layers.append(nn.Linear(curr_layer_input_dim, curr_layer_output_dim))
            layers.append(actiFunc)
            if useBN:
                layers.append(nn.BatchNorm1d(curr_layer_output_dim))               
    else:
        layers = [nn.Flatten(),
                         nn.Linear(linear_input_feature_dim, linear_output_feature_dim),
                         actiFunc]
        if useBN:
            layers += [nn.BatchNorm1d(linear_output_feature_dim)]
    return layers

class NetStrainMat2ClsReg(nn.Module):
    def __init__(self, config={}):
        super(NetStrainMat2ClsReg, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors_in = self.paras.get('n_sectors_in', 18)
        # self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.n_frames = self.paras.get('n_frames', 25)
        self.use_batch_norm = self.paras.get('batch_norm', False)
        self.joint_n_conv_layers = self.paras.get('joint_n_conv_layers', 4)
        self.joint_n_conv_channels = self.paras.get('joint_n_conv_channels', 16)
        self.joint_conv_size = self.paras.get('joint_conv_size', 3)
        self.joint_n_output_channels = self.paras.get('joint_n_output_channels', self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        self.n_classes = self.paras.get('n_classes', 2)

        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size = 1

        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)

        self.joint_conv_layers = nn.ModuleList(
            conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size,
                     joint_stride_size, joint_padding_size, self.joint_n_output_channels,
                     self.n_joint_pooling, actiFunc, useBN=self.use_batch_norm))

        # Regression Conv layers
        self.reg_conv_size = self.paras.get('reg_conv_size', self.joint_conv_size)
        self.reg_n_conv_layers = self.paras.get('reg_n_conv_layers', self.joint_n_conv_layers)
        self.reg_n_conv_channels = self.paras.get('reg_n_conv_channels', self.joint_n_conv_channels)
        self.reg_n_output_channels = self.paras.get('reg_n_output_channels', self.reg_n_conv_channels)
        self.reg_n_linear_layers = self.paras.get('reg_n_linear_layers', 3)
        self.reg_n_dim_out = self.paras.get('reg_n_dim_out', 18)

        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1

        reg_convs = conv_net(self.joint_n_output_channels, self.reg_n_conv_layers, self.reg_n_conv_channels,
                             self.reg_conv_size,
                             reg_conv_stride, reg_padding_size, self.reg_n_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)

        # Regression Linear Layers
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**2))
        reg_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.reg_n_output_channels // (
                    1 * (4 ** self.n_joint_pooling))
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**self.n_joint_pooling/2))
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        reg_linear_output_feature_dim = self.reg_n_dim_out

        reg_linears = fcn(self.reg_n_linear_layers, reg_linear_input_feature_dim, reg_linear_inner_feature_dim,
                          reg_linear_output_feature_dim,
                          actiFunc, useBN=self.use_batch_norm)

        # print(reg_convs, reg_linears)
        self.reg_layers = nn.ModuleList(reg_convs + reg_linears)

        # Classification Conv Layers        
        self.cls_conv_size = self.paras.get('cls_conv_size', self.joint_conv_size)
        self.cls_n_conv_layers = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        self.cls_n_conv_channels = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_n_output_channels = self.paras.get('cls_n_output_channels', self.cls_n_conv_channels)
        self.cls_n_linear_layers = self.paras.get('cls_n_linear_layers', 3)
        self.cls_n_dim_out = self.paras.get('cls_n_dim_out', 18)

        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1

        cls_convs = conv_net(self.joint_n_output_channels, self.cls_n_conv_layers, self.cls_n_conv_channels,
                             self.cls_conv_size,
                             cls_conv_stride, cls_padding_size, self.cls_n_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)

        # Classification Linear Layers
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        cls_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.cls_n_output_channels // (
                    1 * (4 ** self.n_joint_pooling))
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**self.n_joint_pooling/2))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim // 2

        cls_linears = fcn(self.cls_n_linear_layers, cls_linear_input_feature_dim, cls_linear_inner_feature_dim,
                          self.cls_n_dim_out,
                          actiFunc, useBN=self.use_batch_norm)

        self.cls_layers = nn.ModuleList(cls_convs + cls_linears)

    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type_by_category('strainmat', input_types)

    def set_output_types(self, output_types, reg_category='TOS', cls_category='sector_label'):
        self.reg_type = get_data_type_by_category(reg_category, output_types)
        # self.reg_type = get_data_type('fit_coefs', output_types)
        self.cls_type = get_data_type_by_category(cls_category, output_types)

    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        for layer in self.joint_conv_layers:
            x = layer(x)

        x_cls = x
        for layer in self.cls_layers:
            x_cls = layer(x_cls)

        x_cls = x_cls.reshape(x.shape[0], self.n_classes, -1)
        # xls: [N, n_clasees, n_sectors]
        x_cls = nn.Softmax(dim=1)(x_cls)
        # print(x_cls.shape)

        x_reg = x
        for layer in self.reg_layers:
            x_reg = layer(x_reg)
        x_reg = nn.LeakyReLU(inplace=True)(x_reg - 17) + 17
        # x_reg[::2] = torch.round(x_reg[::2])
        return {self.reg_type: x_reg, self.cls_type: x_cls}


class NetStrainMat2ClsDistMapReg(nn.Module):
    def __init__(self, config={}):
        super(NetStrainMat2ClsDistMapReg, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors_in = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out = self.paras.get('n_sectors_out', 18)
        self.n_frames = self.paras.get('n_frames', 25)
        self.use_batch_norm = self.paras.get('batch_norm', False)
        self.joint_n_conv_layers = self.paras.get('joint_n_conv_layers', 4)
        self.joint_n_conv_channels = self.paras.get('joint_n_conv_channels', 16)
        self.joint_conv_size = self.paras.get('joint_conv_size', 3)
        self.joint_n_output_channels = self.paras.get('joint_n_output_channels', self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        self.n_classes = self.paras.get('n_classes', 2)

        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size = 1

        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)

        self.joint_conv_layers = nn.ModuleList(
            conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size,
                     joint_stride_size, joint_padding_size, self.joint_n_output_channels,
                     self.n_joint_pooling, actiFunc, useBN=self.use_batch_norm))

        # Regression Conv layers
        self.reg_conv_size = self.paras.get('reg_conv_size', self.joint_conv_size)
        self.reg_n_conv_layers = self.paras.get('reg_n_conv_layers', self.joint_n_conv_layers)
        self.reg_n_conv_channels = self.paras.get('reg_n_conv_channels', self.joint_n_conv_channels)
        self.reg_n_output_channels = self.paras.get('reg_n_output_channels', self.reg_n_conv_channels)
        self.reg_n_linear_layers = self.paras.get('reg_n_linear_layers', 3)

        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1

        reg_convs = conv_net(self.joint_n_output_channels, self.reg_n_conv_layers, self.reg_n_conv_channels,
                             self.reg_conv_size,
                             reg_conv_stride, reg_padding_size, self.reg_n_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)

        # Regression Linear Layers
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**2))
        reg_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.reg_n_output_channels // (
                    1 * (4 ** self.n_joint_pooling))
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**self.n_joint_pooling/2))
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        reg_linear_output_feature_dim = self.n_sectors_out

        reg_linears = fcn(self.reg_n_linear_layers, reg_linear_input_feature_dim, reg_linear_inner_feature_dim,
                          reg_linear_output_feature_dim,
                          actiFunc, useBN=self.use_batch_norm)

        # print(reg_convs, reg_linears)
        self.reg_layers = nn.ModuleList(reg_convs + reg_linears)

        # Classification Conv Layers        
        self.cls_conv_size = self.paras.get('cls_conv_size', self.joint_conv_size)
        self.cls_n_conv_layers = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        self.cls_n_conv_channels = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_n_output_channels = self.paras.get('cls_n_output_channels', self.cls_n_conv_channels)
        self.cls_n_linear_layers = self.paras.get('cls_n_linear_layers', 3)
        cls_bnFunc_conv_inner = nn.BatchNorm2d(self.cls_n_conv_channels) if useBN else nn.Identity()
        cls_bnFunc_conv_output = nn.BatchNorm2d(self.cls_n_output_channels) if useBN else nn.Identity()

        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1

        cls_convs = conv_net(self.joint_n_output_channels, self.cls_n_conv_layers, self.cls_n_conv_channels,
                             self.cls_conv_size,
                             cls_conv_stride, cls_padding_size, self.cls_n_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)

        # Classification Linear Layers
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        cls_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.cls_n_output_channels // (
                    1 * (4 ** self.n_joint_pooling))
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**self.n_joint_pooling/2))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim // 2

        cls_linears = fcn(self.cls_n_linear_layers, cls_linear_input_feature_dim, cls_linear_inner_feature_dim,
                          self.n_classes * self.n_sectors_out,
                          actiFunc, useBN=self.use_batch_norm)

        self.cls_layers = nn.ModuleList(cls_convs + cls_linears)

    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type_by_category('strainmat', input_types)

    def set_output_types(self, output_types):
        self.TOS_type = get_data_type_by_category('TOS', output_types)
        self.classify_label_type = get_data_type_by_category('sector_dist_map', output_types)

    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        for layer in self.joint_conv_layers:
            x = layer(x)

        x_cls = x
        for layer in self.cls_layers:
            x_cls = layer(x_cls)

        x_cls = x_cls.reshape(-1, self.n_classes, self.n_sectors_out)

        x_reg = x
        for layer in self.reg_layers:
            x_reg = layer(x_reg)
        x_reg = nn.LeakyReLU(inplace=True)(x_reg - 17) + 17
        return {self.TOS_type: x_reg, self.classify_label_type: x_cls}


# def conv_pooling_bn_acti_layer(conv_in_channels, conv_out_channels, conv_size, conv_stride, conv_padding,
#                                actiFunc, useBN=False):
#     layers = [
#         nn.Conv2d(conv_in_channels, conv_out_channels, conv_size,
#                   stride=conv_stride, padding=conv_padding),
#     ]
#     if useBN:
#         layers += [nn.BatchNorm2d(joint_n_output_channels)]
#     layers += [
#         actiFunc,
#         nn.MaxPool2d((1, 2), stride=(1, 2))
#     ]
#     return layers


class NetStrainMatGeoEncoder(nn.Module):
    def __init__(self, config={}):
        super(NetStrainMatGeoEncoder, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors_in = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out = self.paras.get('n_sectors_out', 18)
        self.n_frames = self.paras.get('n_frames', 25)
        self.use_batch_norm = self.paras.get('batch_norm', False)
        self.joint_n_conv_layers = self.paras.get('joint_n_conv_layers', 4)
        self.joint_n_conv_channels = self.paras.get('joint_n_conv_channels', 16)
        self.joint_conv_size = self.paras.get('joint_conv_size', 3)
        self.joint_n_output_channels = self.paras.get('joint_n_output_channels', self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        self.n_classes = self.paras.get('n_classes', 2)

        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size = 1

        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)

        self.joint_conv_layers = nn.ModuleList(
            conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size,
                     joint_stride_size, joint_padding_size, self.joint_n_output_channels,
                     self.n_joint_pooling, actiFunc, useBN=self.use_batch_norm))

        # Regression Conv layers
        self.reg_conv_size = self.paras.get('reg_conv_size', self.joint_conv_size)
        self.reg_n_conv_layers = self.paras.get('reg_n_conv_layers', self.joint_n_conv_layers)
        self.reg_n_conv_channels = self.paras.get('reg_n_conv_channels', self.joint_n_conv_channels)
        self.reg_n_output_channels = self.paras.get('reg_n_output_channels', self.reg_n_conv_channels)
        self.reg_n_linear_layers = self.paras.get('reg_n_linear_layers', 3)

        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1

        reg_convs = conv_net(self.joint_n_output_channels, self.reg_n_conv_layers, self.reg_n_conv_channels,
                             self.reg_conv_size,
                             reg_conv_stride, reg_padding_size, self.reg_n_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)

        # Regression Linear Layers
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**2))
        reg_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.reg_n_output_channels // (
                    1 * (4 ** self.n_joint_pooling))
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**self.n_joint_pooling/2))
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        reg_linear_output_feature_dim = self.n_sectors_out

        reg_linears = fcn(self.reg_n_linear_layers, reg_linear_input_feature_dim, reg_linear_inner_feature_dim,
                          reg_linear_output_feature_dim,
                          actiFunc, useBN=self.use_batch_norm)

        # print(reg_convs, reg_linears)
        self.reg_layers = nn.ModuleList(reg_convs + reg_linears)

        # Classification Conv Layers        
        self.cls_conv_size = self.paras.get('cls_conv_size', self.joint_conv_size)
        self.cls_n_conv_layers = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        self.cls_n_conv_channels = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_n_output_channels = self.paras.get('cls_n_output_channels', self.cls_n_conv_channels)
        self.cls_n_linear_layers = self.paras.get('cls_n_linear_layers', 3)
        cls_bnFunc_conv_inner = nn.BatchNorm2d(self.cls_n_conv_channels) if useBN else nn.Identity()
        cls_bnFunc_conv_output = nn.BatchNorm2d(self.cls_n_output_channels) if useBN else nn.Identity()

        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1

        cls_convs = conv_net(self.joint_n_output_channels, self.cls_n_conv_layers, self.cls_n_conv_channels,
                             self.cls_conv_size,
                             cls_conv_stride, cls_padding_size, self.cls_n_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)
        

        # Classification Linear Layers
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        cls_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.cls_n_output_channels // (
                    1 * (4 ** self.n_joint_pooling))
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**self.n_joint_pooling/2))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim // 2

        cls_linears = fcn(self.cls_n_linear_layers, cls_linear_input_feature_dim, cls_linear_inner_feature_dim,
                          self.n_classes * self.n_sectors_out,
                          actiFunc, useBN=self.use_batch_norm)

        self.cls_layers = nn.ModuleList(cls_convs + cls_linears)

    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type_by_category('strainmat', input_types)

    def set_output_types(self, output_types, reg_category='TOS', cls_category='sector_dist_map'):
        self.reg_type = get_data_type_by_category(reg_category, output_types)
        self.cls_type = get_data_type_by_category(cls_category, output_types)

    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        for layer in self.joint_conv_layers:
            x = layer(x)

        x_cls = x
        for layer in self.cls_layers:
            x_cls = layer(x_cls)

        x_cls = x_cls.reshape(-1, self.n_classes, self.n_sectors_out)

        x_reg = x
        for layer in self.reg_layers:
            x_reg = layer(x_reg)
        x_reg = nn.LeakyReLU(inplace=True)(x_reg - 17) + 17
        return {self.reg_type: x_reg, self.cls_type: x_cls}


# class NetStrainMat2LateActiCls(nn.Module):
class NetStrainMat2Cls(nn.Module):
    def __init__(self, config={}):
        # super(NetStrainMat2Cls, self).__init__()
        super().__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors_in = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out = self.paras.get('n_sectors_out', 18)
        self.n_frames = self.paras.get('n_frames', 25)
        self.n_conv_layers = self.paras.get('n_conv_layers', 4)        
        self.n_conv_channels = self.paras.get('n_conv_channels', 16)
        self.n_conv_output_channels = self.paras.get('n_conv_output_channels', 2)
        self.conv_size = self.paras.get('conv_size', 3)
        self.conv_stride = self.paras.get('conv_stride', 1)
        self.n_linear_layers = self.paras.get('n_linear_layers', 3)
        self.cls_n_dim_out = self.paras.get('cls_n_dim_out', 2)
        self.n_classes = self.paras.get('n_clasaes', 2)
        self.activation_func = self.paras.get('activation_func', 'ReLU')

        # useBN = False
        self.use_batch_norm = self.paras.get('batch_norm', True)
        if self.activation_func.lower() == 'relu':
            actiFunc = nn.ReLU(True)
        elif self.activation_func.lower() == 'leaky_relu':
            actiFunc = nn.LeakyReLU(True)
        elif self.activation_func.lower() == 'sigmoid':
            actiFunc = nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation function: ', self.activation_func)
        
        conv_padding_size = self.conv_size // 2
        convs = conv_net(self.n_input_channels, self.n_conv_layers, self.n_conv_channels,
                             self.conv_size,
                             self.conv_stride, conv_padding_size, self.n_conv_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)
        
        # def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc, useBN=False)
        linear_input_dim = self.n_sectors_in * self.n_frames * self.n_conv_output_channels
        # linear = fcn(self.n_linear_layers, linear_input_dim, linear_input_dim // 2, self.cls_n_dim_out, actiFunc,
        linear = fcn(self.n_linear_layers, linear_input_dim, None, self.cls_n_dim_out, actiFunc,
        useBN=True)
        self.layers = nn.ModuleList(convs + linear)

    @autocast()
    def forward(self, input_data: dict or torch.Tensor):
        if type(input_data) is dict:
            x = input_data[self.input_type]  # .cuda()
        else:
            x = input_data
        for layer in self.layers:
            x = layer(x)
        
        if self.n_classes > 2:
            x = x.reshape(x.shape[0], self.n_classes, -1)
            # xls: [N, n_clasees, n_sectors]
            x = nn.Softmax(dim=1)(x)
        else:
            x = nn.LeakyReLU(inplace=True)(x)
            # x = nn.ReLU(inplace=True)(x)
            # pass
        # return x
        
        if type(input_data) is dict:
            return {self.output_types: x}
        else:
            return x

    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.input_type = input_types
        elif type(input_types) is list:
            self.input_type = get_data_type_by_category('strainmat', input_types)

    def set_output_types(self, output_types):
        if type(output_types) is str:
            self.output_types = output_types
        elif type(output_types) is list:
            sector_label_type = get_data_type_by_category('sector_label', output_types)
            if sector_label_type is not None:
                self.output_types = get_data_type_by_category('sector_label', output_types)            
            else:
                self.output_types = get_data_type_by_category('data_label', output_types)            

class shift_Leaky_ReLU(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, shift):
        super().__init__()   
        self.shift = shift

    def forward(self, x):        
        return nn.LeakyReLU(inplace=True)(x - self.shift) + self.shift
            
class NetStrainMat2Reg(nn.Module):
    def __init__(self, config={}):
        # super(NetStrainMat2Reg, self).__init__()
        super().__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors_in = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out = self.paras.get('n_sectors_out', 18)
        self.n_frames = self.paras.get('n_frames', 25)
        self.n_conv_layers = self.paras.get('n_conv_layers', 4)        
        self.n_conv_channels = self.paras.get('n_conv_channels', 16)
        self.n_conv_output_channels = self.paras.get('n_conv_output_channels', 2)
        self.conv_size = self.paras.get('conv_size', 3)
        self.conv_stride = self.paras.get('conv_stride', 1)
        self.n_linear_layers = self.paras.get('n_linear_layers', 3)
        self.reg_n_dim_out = self.paras.get('reg_n_dim_out', 2)
        self.n_classes = self.paras.get('n_clasaes', 1)
        self.add_last_relu = self.paras.get('add_last_relu', False)
        self.activation_func = self.paras.get('activation_func', 'ReLU')

        # useBN = False
        self.use_batch_norm = self.paras.get('batch_norm', True)
        if self.activation_func.lower() == 'relu':
            actiFunc = nn.ReLU(True)
        elif self.activation_func.lower() == 'sigmoid':
            actiFunc = nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation function: ', self.activation_func)
        # actiFunc = nn.ReLU(True)
        # actiFunc = nn.LeakyReLU(True)
        
        conv_padding_size = self.conv_size // 2
        convs = conv_net(self.n_input_channels, self.n_conv_layers, self.n_conv_channels,
                             self.conv_size,
                             self.conv_stride, conv_padding_size, self.n_conv_output_channels,
                             0, actiFunc, useBN=self.use_batch_norm)
        
        # def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc, useBN=False)
        linear_input_dim = self.n_sectors_in * self.n_frames * self.n_conv_output_channels
        linear = fcn(self.n_linear_layers, linear_input_dim, linear_input_dim // 2, self.reg_n_dim_out, actiFunc,
        useBN=True)
        
        if self.add_last_relu:
            last_relu = [shift_Leaky_ReLU(17)]
        else:
            last_relu = []
        
        self.layers = nn.ModuleList(convs + linear + last_relu)

    @autocast()
    def forward(self, input_data: dict or torch.Tensor):        
        if type(input_data) is torch.Tensor:
            x = input_data
        else:        
            x = input_data[self.input_type]  # .cuda()
        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(x.shape[0], self.n_classes, -1)
        # x = x.reshape(x.shape[0], -1)
        # xls: [N, n_clasees, n_sectors]
        # return x
        if type(input_data) is torch.Tensor:
            return x
        else:
            return {self.output_types: x}
    # def forward(self, input_dict):        
    #     x = input_dict[self.input_type]  # .cuda()
    #     for layer in self.layers:
    #         x = layer(x)
    #     x = x.reshape(x.shape[0], self.n_classes, -1)
    #     # xls: [N, n_clasees, n_sectors]
    #     # return x
    #     return {self.output_types: x}
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.input_type = input_types
        elif type(input_types) is list:
            self.input_type = get_data_type_by_category('strainmat', input_types)

    def set_output_types(self, output_types):
        if type(output_types) is str:
            self.output_types = output_types
        elif type(output_types) is list:
            try:
                self.output_types = get_data_type_by_category('sector_dist_map', output_types)                     
            except:
                # print('failed')
                self.output_types = get_data_type_by_category('sector_value', output_types)     
                # print(self.output_types)