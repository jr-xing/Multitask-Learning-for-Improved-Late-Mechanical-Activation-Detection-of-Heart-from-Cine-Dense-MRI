# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:37:37 2021

@author: remus
"""

import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
from utils.data import get_data_type
# from modules.networks.Siamese import NetSiamese, NetSimpleFCN
def get_network_by_name(name, config = {}):
    # if name =='NetStrainMatSectionClassify':        
    #     net = NetStrainMatSectionClassify(config)
    # elif name =='NetStrainMatJointClsPred':
    #     net = NetStrainMatJointClsPred(config)
    # elif name =='NetStrainMat2TOS':
    #     net = NetStrainMat2TOS(config)
    if name =='NetStrainMat2TOS':
        net = NetStrainMat2TOS(config)
    elif name == 'NetStrainMat2ClsReg':
        net =  NetStrainMat2ClsReg(config)
    elif name == 'NetStrainMat2ClsDistMapReg':
        net =  NetStrainMat2ClsDistMapReg(config)
    else:
        raise ValueError('Unsupported net name: ', name)
    return net

class NetStrainMat2TOS(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2TOS, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.n_input_channels   = self.paras.get('n_input_channels', 1)
        self.n_sectors_in       = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out      = self.paras.get('n_sectors_out', 18)
        self.n_frames           = self.paras.get('n_frames',  25)
        self.n_conv_layers      = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels    = self.paras.get('n_conv_channels',  16)
        self.conv_size          = self.paras.get('conv_size',  3)
        self.conv_stride        = self.paras.get('conv_stride', 1)
        
        useBN = False
        # actiFunc = nn.Sigmoid()
        actiFunc = nn.ReLU(True)
        # actiFunc = nn.LeakyReLU(True)
        bnFuncConv1 = nn.BatchNorm2d(self.n_conv_channels) if useBN else nn.Identity()
        bnFuncConv2 = nn.BatchNorm2d(1) if useBN else nn.Identity()
        convs = [nn.Conv2d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=self.conv_stride, padding=self.conv_size//2), actiFunc, bnFuncConv1]
        for innerLayerIdx in range(self.n_conv_layers-2):
            convs += [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=self.conv_stride, padding=self.conv_size//2), actiFunc, bnFuncConv1]
        convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=self.conv_stride, padding=self.conv_size//2), actiFunc, bnFuncConv2]
        
        
        # linear = [nn.Flatten(), nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_out)]
        
        # linear = [nn.Flatten(), \
        #           nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_in*self.n_frames//2),\
        #           bnFunc, actiFunc, \
        #           nn.Linear(self.n_sectors_in*self.n_frames//2, self.n_sectors_out)]
        bnFuncLinear1 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//2) if useBN else nn.Identity()
        bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        linear = [nn.Flatten(), \
                  nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_in*self.n_frames//2),\
                  actiFunc,bnFuncLinear1,  \
                  nn.Linear(self.n_sectors_in*self.n_frames//2, self.n_sectors_in*self.n_frames//4),
                  actiFunc,bnFuncLinear2,  \
                  nn.Linear(self.n_sectors_in*self.n_frames//4, self.n_sectors_out)]
        self.layers = nn.ModuleList(convs+linear)
        
    @autocast()
    def forward(self, input_dict):
        # with autocast():
        x = input_dict[self.input_type]#.cuda()
        # print(self.input_type, x.dtype)
        # x = self.net(x)        
        for layer in self.layers:
            # print(x.shape)
            # print(layer)
            # print(layer.weight.type())
            x = layer(x)        
        x = nn.LeakyReLU(inplace = True)(x-17)+17
        # return x
        return {self.output_types: x}
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.input_type = input_types
        elif type(input_types) is list:
            self.input_type = get_data_type('strainmat', input_types)
    
    def set_output_types(self, output_types):
        if type(output_types) is str:
            self.output_types = output_types
        elif type(output_types) is list:
            self.output_types = get_data_type('TOS', output_types)


def joint_conv_net(n_input_channels, joint_n_conv_layers, joint_n_conv_channels, joint_conv_size, 
                   joint_stride_size, joint_padding_size, joint_n_output_channels,
                   n_joint_pooling, actiFunc,useBN = False):
    # useBN = self.use_batch_norm
    # actiFunc = nn.Sigmoid()
    # actiFunc = nn.ReLU(True)
    actiFunc = nn.LeakyReLU(True)
    
    joint_convs = [nn.Conv2d(n_input_channels, joint_n_conv_channels, joint_conv_size, 
                       stride=joint_stride_size, padding=joint_padding_size), actiFunc]
    if useBN:
        joint_convs += [nn.BatchNorm2d(joint_n_conv_channels)]
    
    if n_joint_pooling > 1:
        joint_convs += [nn.MaxPool2d(2, stride=2)]
        # joint_convs += [nn.MaxPool2d((1,2), stride = (1,2))]
    
    for innerLayerIdx in range(joint_n_conv_layers-2):
        joint_convs += [nn.Conv2d(joint_n_conv_channels, joint_n_conv_channels, joint_conv_size, 
                            stride = joint_stride_size, padding = joint_padding_size), 
                        actiFunc]
        if useBN:
            joint_convs += [nn.BatchNorm2d(joint_n_conv_channels)]
    joint_convs += [nn.Conv2d(joint_n_conv_channels, joint_n_output_channels, joint_conv_size, 
                        stride = joint_stride_size, padding = joint_padding_size), 
              actiFunc]
    if useBN:
        joint_convs += [nn.BatchNorm2d(joint_n_output_channels)]
        
    if n_joint_pooling > 0:
        joint_convs += [nn.MaxPool2d(2, stride=2)]
    
    return nn.ModuleList(joint_convs)
            
class NetStrainMat2ClsReg(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2ClsReg, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels       = self.paras.get('n_input_channels', 1)
        self.n_sectors_in           = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.n_frames               = self.paras.get('n_frames',  25)
        self.use_batch_norm         = self.paras.get('batch_norm',  False)
        self.joint_n_conv_layers    = self.paras.get('joint_n_conv_layers',  4)
        self.joint_n_conv_channels  = self.paras.get('joint_n_conv_channels',  16)
        self.joint_conv_size        = self.paras.get('joint_conv_size',  3)
        self.joint_n_output_channels= self.paras.get('joint_n_output_channels',  self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        
        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size  = 1
        
        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)
        
        joint_convs = [nn.Conv2d(self.n_input_channels, self.joint_n_conv_channels, self.joint_conv_size, 
                           stride=joint_stride_size, padding=joint_padding_size), actiFunc]
        if useBN:
            joint_convs += [nn.BatchNorm2d(self.joint_n_conv_channels)]
        
        if self.n_joint_pooling > 1:
            joint_convs += [nn.MaxPool2d(2, stride=2)]
            # joint_convs += [nn.MaxPool2d((1,2), stride = (1,2))]
        
        for innerLayerIdx in range(self.joint_n_conv_layers-2):
            joint_convs += [nn.Conv2d(self.joint_n_conv_channels, self.joint_n_conv_channels, self.joint_conv_size, 
                                stride = joint_stride_size, padding = joint_padding_size), 
                            actiFunc]
            if useBN:
                joint_convs += [nn.BatchNorm2d(self.joint_n_conv_channels)]
        joint_convs += [nn.Conv2d(self.joint_n_conv_channels, self.joint_n_output_channels, self.joint_conv_size, 
                            stride = joint_stride_size, padding = joint_padding_size), 
                  actiFunc]
        if useBN:
            joint_convs += [nn.BatchNorm2d(self.joint_n_output_channels)]
            
        if self.n_joint_pooling > 0:
            joint_convs += [nn.MaxPool2d(2, stride=2)]
        
        self.joint_conv_layers = joint_conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size, 
                   self.joint_stride_size, self.joint_padding_size, self.joint_n_output_channels,
                   self.n_joint_pooling, actiFunc,useBN = self.use_batch_norm)
        
        
        # convs += [nn.MaxPool2d((1,2), stride = (1,2))]
        # convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), bnFunc, actiFunc]
        
        # Regression Conv layers
        self.reg_conv_size         = self.paras.get('reg_conv_size', self.joint_conv_size)
        self.reg_n_conv_layers     = self.paras.get('reg_n_conv_layers', self.joint_n_conv_layers)
        self.reg_n_conv_channels   = self.paras.get('reg_n_conv_channels', self.joint_n_conv_channels)
        self.reg_n_output_channels = self.paras.get('reg_n_output_channels',  self.reg_n_conv_channels)
        self.reg_n_linear_layers   = self.paras.get('reg_n_linear_layers', 3)
        # reg_bnFunc_conv_inner = nn.BatchNorm2d(self.reg_n_conv_channels) if useBN else nn.Identity()
        # reg_bnFunc_conv_output = nn.BatchNorm2d(self.reg_n_output_channels) if useBN else nn.Identity()
        
        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1
        
        reg_convs = [nn.Conv2d(self.joint_n_output_channels, self.reg_n_conv_channels, self.reg_conv_size, 
                           stride=reg_conv_stride, padding=reg_padding_size), 
                     actiFunc]
        if useBN:
            reg_convs += [nn.BatchNorm2d(self.reg_n_conv_channels)]
        for innerLayerIdx in range(self.reg_n_conv_layers-2):
            reg_convs += [nn.Conv2d(self.reg_n_conv_channels, self.reg_n_conv_channels, self.reg_conv_size, 
                                stride = reg_conv_stride, padding = reg_padding_size), 
                          actiFunc]
        if useBN:
            reg_convs += [nn.BatchNorm2d(self.reg_n_conv_channels)]
        reg_convs += [nn.Conv2d(self.reg_n_conv_channels, self.reg_n_output_channels, self.reg_conv_size, 
                            stride=reg_conv_stride, padding=reg_padding_size), actiFunc]
        if useBN:
            reg_convs += [nn.BatchNorm2d(self.reg_n_output_channels)]
        
        # Regression Linear Layers
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**2))
        reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**self.n_joint_pooling))
        # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**self.n_joint_pooling/2))
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        reg_bnFunc_linear_inner = nn.BatchNorm1d(reg_linear_inner_feature_dim) if useBN else nn.Identity()
        # bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        reg_linears = [nn.Flatten(), 
                       nn.Linear(reg_linear_input_feature_dim, reg_linear_inner_feature_dim),
                       actiFunc, reg_bnFunc_linear_inner]
        for inner_layer_idx in range(self.reg_n_linear_layers - 2):
            reg_linears += [nn.Linear(reg_linear_inner_feature_dim, reg_linear_inner_feature_dim), actiFunc, reg_bnFunc_linear_inner]
        reg_linears += [nn.Linear(reg_linear_inner_feature_dim, self.n_sectors_out)]
        
        self.reg_layers = nn.ModuleList(reg_convs + reg_linears)
        
        # Classification Conv Layers
        self.n_classes             = self.paras.get('n_classes', 2)
        self.cls_conv_size         = self.paras.get('cls_conv_size', self.joint_conv_size)
        self.cls_n_conv_layers     = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        self.cls_n_conv_channels   = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_n_output_channels = self.paras.get('cls_n_output_channels',  self.cls_n_conv_channels)
        self.cls_n_linear_layers   = self.paras.get('cls_n_linear_layers', 3)
        cls_bnFunc_conv_inner = nn.BatchNorm2d(self.cls_n_conv_channels) if useBN else nn.Identity()
        cls_bnFunc_conv_output = nn.BatchNorm2d(self.cls_n_output_channels) if useBN else nn.Identity()
        
        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1
        
        cls_convs = [nn.Conv2d(self.joint_n_output_channels, self.cls_n_conv_channels, self.cls_conv_size, 
                           stride=cls_conv_stride, padding=cls_padding_size), 
                     actiFunc, cls_bnFunc_conv_inner]
        for innerLayerIdx in range(self.cls_n_conv_layers-2):
            cls_convs += [nn.Conv2d(self.cls_n_conv_channels, self.cls_n_conv_channels, self.cls_conv_size, 
                                stride = cls_conv_stride, padding = cls_padding_size), 
                          actiFunc, cls_bnFunc_conv_inner]
        cls_convs += [nn.Conv2d(self.cls_n_conv_channels, self.cls_n_output_channels, self.cls_conv_size, 
                            stride=cls_conv_stride, padding=cls_padding_size), 
                        actiFunc, cls_bnFunc_conv_output]
        
        
        # Classification Linear Layers
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (1*(4**self.n_joint_pooling))
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**self.n_joint_pooling/2))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim // 2
        cls_bnFunc_linear_inner = nn.BatchNorm1d(cls_linear_inner_feature_dim) if useBN else nn.Identity()
        # bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        cls_linears = [nn.Flatten(), 
                       nn.Linear(cls_linear_input_feature_dim, cls_linear_inner_feature_dim),
                       actiFunc, cls_bnFunc_linear_inner]
        for inner_layer_idx in range(self.cls_n_linear_layers - 2):
            cls_linears += [nn.Linear(cls_linear_inner_feature_dim, cls_linear_inner_feature_dim), actiFunc, cls_bnFunc_linear_inner]
        cls_linears += [nn.Linear(cls_linear_inner_feature_dim, self.n_classes*self.n_sectors_out)]        
        
        self.cls_layers = nn.ModuleList(cls_convs + cls_linears)                
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type('strainmat', input_types)
    
    def set_output_types(self, output_types):
        self.TOS_type = get_data_type('TOS', output_types)
        self.classify_label_type = get_data_type('sector_label', output_types)
    
    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        # x = self.net(x)        
        # raise ValueError('STOP!')
        # print('init x: ', x.shape)
        for layer in self.joint_conv_layers:
            x = layer(x)
        
        # for layer in self.joint_linear_layers:
        #     x = layer(x)
        # print('After conv layers: ', x.shape)        
        x_cls = x
        for layer in self.cls_layers:
            # print(x_cls.shape)
            # print(layer)
            x_cls = layer(x_cls)        
        # x = nn.LeakyReLU(inplace = True)(x-17)+17
        # print(x.shape)
        x_cls = x_cls.reshape(-1, self.n_classes, self.n_sectors_out)
        # xls: [N, n_clasees, n_sectors]
        x_cls = nn.Softmax(dim = 1)(x_cls)
        # print(x.shape)
        
        x_reg = x
        for layer in self.reg_layers:
            # print(x_reg.shape)
            # print(layer)            
            x_reg = layer(x_reg)        
        # x = nn.LeakyReLU(inplace = True)(x-17)+17
        # print(x.shape)
        x_reg = nn.LeakyReLU(inplace = True)(x_reg-17)+17
        # print('x_reg: ', x_reg.shape)
        # print('FWD!')
        return {self.TOS_type: x_reg, self.classify_label_type: x_cls}

class NetStrainMat2ClsDistMapReg(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2ClsDistMapReg, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels       = self.paras.get('n_input_channels', 1)
        self.n_sectors_in           = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.n_frames               = self.paras.get('n_frames',  25)
        self.use_batch_norm         = self.paras.get('batch_norm',  False)
        self.joint_n_conv_layers    = self.paras.get('joint_n_conv_layers',  4)
        self.joint_n_conv_channels  = self.paras.get('joint_n_conv_channels',  16)
        self.joint_conv_size        = self.paras.get('joint_conv_size',  3)
        self.joint_n_output_channels= self.paras.get('joint_n_output_channels',  self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        
        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size  = 1
        
        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)
        joint_bnFunc_conv_inner = nn.BatchNorm2d(self.joint_n_conv_channels) if useBN else nn.Identity()
        joint_bnFunc_conv_output = nn.BatchNorm2d(self.joint_n_output_channels) if useBN else nn.Identity()
        
        joint_convs = [nn.Conv2d(self.n_input_channels, self.joint_n_conv_channels, self.joint_conv_size, 
                           stride=joint_stride_size, padding=joint_padding_size), actiFunc, joint_bnFunc_conv_inner]        
        if self.n_joint_pooling > 0:
            # joint_convs += [nn.MaxPool2d((1,2), stride = (1,2))]
            joint_convs += [nn.MaxPool2d((2,2), stride = (2,2))]
        for innerLayerIdx in range(self.joint_n_conv_layers-2):
            joint_convs += [nn.Conv2d(self.joint_n_conv_channels, self.joint_n_conv_channels, self.joint_conv_size, 
                                stride = joint_stride_size, padding = joint_padding_size), 
                            actiFunc, joint_bnFunc_conv_inner]            
        joint_convs += [nn.Conv2d(self.joint_n_conv_channels, self.joint_n_output_channels, self.joint_conv_size, 
                            stride = joint_stride_size, padding = joint_padding_size), 
                  actiFunc, joint_bnFunc_conv_output]
        if self.n_joint_pooling > 1:
            joint_convs += [nn.MaxPool2d(2, stride=2)]
        
        self.joint_conv_layers = nn.ModuleList(joint_convs)
        
        
        # convs += [nn.MaxPool2d((1,2), stride = (1,2))]
        # convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), bnFunc, actiFunc]
        
        # Regression Conv layers
        self.reg_conv_size         = self.paras.get('reg_conv_size', self.joint_conv_size)
        self.reg_n_conv_layers     = self.paras.get('reg_n_conv_layers', self.joint_n_conv_layers)
        # self.reg_n_conv_layers     = 1
        self.reg_n_conv_channels   = self.paras.get('reg_n_conv_channels', self.joint_n_conv_channels)
        self.reg_n_output_channels = self.paras.get('reg_n_output_channels',  self.reg_n_conv_channels)
        self.reg_n_linear_layers   = self.paras.get('reg_n_linear_layers', 3)
        reg_bnFunc_conv_inner = nn.BatchNorm2d(self.reg_n_conv_channels) if useBN else nn.Identity()
        reg_bnFunc_conv_output = nn.BatchNorm2d(self.reg_n_output_channels) if useBN else nn.Identity()
        
        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1
        
        if self.reg_n_conv_layers >= 2:
            reg_convs = [nn.Conv2d(self.joint_n_output_channels, self.reg_n_conv_channels, self.reg_conv_size, 
                            stride=reg_conv_stride, padding=reg_padding_size), 
                        actiFunc, reg_bnFunc_conv_inner]
            for innerLayerIdx in range(self.reg_n_conv_layers-2):
                reg_convs += [nn.Conv2d(self.reg_n_conv_channels, self.reg_n_conv_channels, self.reg_conv_size, 
                                    stride = reg_conv_stride, padding = reg_padding_size), 
                            actiFunc, reg_bnFunc_conv_inner]
            reg_convs += [nn.Conv2d(self.reg_n_conv_channels, self.reg_n_output_channels, self.reg_conv_size, 
                                stride=reg_conv_stride, padding=reg_padding_size), 
                            actiFunc, reg_bnFunc_conv_output]
        elif self.reg_n_conv_layers == 1:
            reg_convs = [nn.Conv2d(self.joint_n_output_channels, self.reg_n_output_channels, self.reg_conv_size, 
                            stride=reg_conv_stride, padding=reg_padding_size), 
                        actiFunc, reg_bnFunc_conv_output]
        elif self.reg_n_conv_layers == 0:
            reg_convs = [nn.Identity()]
            self.reg_n_output_channels = self.joint_n_output_channels
        
        
        # Regression Linear Layers
        reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**self.n_joint_pooling))        
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        # if self.n_joint_pooling > 0:
        #     reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**(self.n_joint_pooling-1)))            
        #     reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**(self.n_joint_pooling-1)))            
        # else:
        #     reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels
        #     reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels

        reg_bnFunc_linear_inner = nn.BatchNorm1d(reg_linear_inner_feature_dim) if useBN else nn.Identity()
        # bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        reg_linears = [nn.Flatten(), 
                       nn.Linear(reg_linear_input_feature_dim, reg_linear_inner_feature_dim),
                       actiFunc, reg_bnFunc_linear_inner]
        for inner_layer_idx in range(self.reg_n_linear_layers - 2):
            reg_linears += [nn.Linear(reg_linear_inner_feature_dim, reg_linear_inner_feature_dim), actiFunc, reg_bnFunc_linear_inner]
        reg_linears += [nn.Linear(reg_linear_inner_feature_dim, self.n_sectors_out)]
        
        self.reg_layers = nn.ModuleList(reg_convs + reg_linears)
        
        # Classification Conv Layers
        self.n_classes             = self.paras.get('n_classes', 1)
        self.cls_conv_size         = self.paras.get('cls_conv_size', self.joint_conv_size)
        self.cls_n_conv_layers     = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        # self.cls_n_conv_layers     = 0
        self.cls_n_conv_channels   = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_n_output_channels = self.paras.get('cls_n_output_channels',  self.cls_n_conv_channels)
        self.cls_n_linear_layers   = self.paras.get('cls_n_linear_layers', 3)
        cls_bnFunc_conv_inner = nn.BatchNorm2d(self.cls_n_conv_channels) if useBN else nn.Identity()
        cls_bnFunc_conv_output = nn.BatchNorm2d(self.cls_n_output_channels) if useBN else nn.Identity()
        
        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1
        
        if self.cls_n_conv_layers >= 2:
            cls_convs = [nn.Conv2d(self.joint_n_output_channels, self.cls_n_conv_channels, self.cls_conv_size, 
                            stride=cls_conv_stride, padding=cls_padding_size), 
                        actiFunc, cls_bnFunc_conv_inner]
            for innerLayerIdx in range(self.cls_n_conv_layers-2):
                cls_convs += [nn.Conv2d(self.cls_n_conv_channels, self.cls_n_conv_channels, self.cls_conv_size, 
                                    stride = cls_conv_stride, padding = cls_padding_size), 
                            actiFunc, cls_bnFunc_conv_inner]
            cls_convs += [nn.Conv2d(self.cls_n_conv_channels, self.cls_n_output_channels, self.cls_conv_size, 
                                stride=cls_conv_stride, padding=cls_padding_size), 
                            actiFunc, cls_bnFunc_conv_output]
        elif self.cls_n_conv_layers == 1:
            cls_convs = [nn.Conv2d(self.joint_n_output_channels, self.cls_n_output_channels, self.cls_conv_size, 
                            stride=cls_conv_stride, padding=cls_padding_size), 
                        actiFunc, cls_bnFunc_conv_output]
        elif self.cls_n_conv_layers == 0:
            cls_convs = [nn.Identity()]
            self.cls_n_output_channels = self.joint_n_output_channels
        
        # Classification Linear Layers
        cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (1*(4**self.n_joint_pooling))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim //2
        # reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**self.n_joint_pooling//2))        
        # reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        # if self.n_joint_pooling > 0:
        #     cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (1*(4**2))
        #     cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**2))
        # else:
        #     cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels
        #     cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels

        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        cls_bnFunc_linear_inner = nn.BatchNorm1d(cls_linear_inner_feature_dim) if useBN else nn.Identity()
        # bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        cls_linears = [nn.Flatten(), 
                       nn.Linear(cls_linear_input_feature_dim, cls_linear_inner_feature_dim),
                       actiFunc, cls_bnFunc_linear_inner]
        for inner_layer_idx in range(self.cls_n_linear_layers - 2):
            cls_linears += [nn.Linear(cls_linear_inner_feature_dim, cls_linear_inner_feature_dim), actiFunc, cls_bnFunc_linear_inner]
        cls_linears += [nn.Linear(cls_linear_inner_feature_dim, self.n_classes*self.n_sectors_out)]        
        
        self.cls_layers = nn.ModuleList(cls_convs + cls_linears)                
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type('strainmat', input_types)
    
    def set_output_types(self, output_types):
        self.TOS_type = get_data_type('TOS', output_types)
        self.classify_label_type = get_data_type('sector_dist_map', output_types)
    
    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        # x = self.net(x)        
        # raise ValueError('STOP!')
        # print('init x: ', x.shape)
        for layer in self.joint_conv_layers:
            x = layer(x)
        
        # for layer in self.joint_linear_layers:
        #     x = layer(x)
        # print('After conv layers: ', x.shape)        
        x_cls = x
        for layer in self.cls_layers:
            # print(x_cls.shape)
            # print(layer)
            x_cls = layer(x_cls)
        x_cls = x_cls.view(x_cls.shape[0], self.n_classes, -1)
        # print(x_cls.shape)
        # x_cls = torch.reshape(x_cls, (-1,2,-1))
        
        x_reg = x
        for layer in self.reg_layers:
            # print(x_reg.shape)
            # print(layer)            
            x_reg = layer(x_reg)        
        # x = nn.LeakyReLU(inplace = True)(x-17)+17
        # print(x.shape)
        x_reg = nn.LeakyReLU(inplace = True)(x_reg-17)+17
        # print('x_reg: ', x_reg.shape)
        # print('FWD!')
        return {self.TOS_type: x_reg, self.classify_label_type: x_cls}


class NetStrainMatGeoEncoder(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMatGeoEncoder, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels       = self.paras.get('n_input_channels', 1)
        self.n_sectors_in           = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.n_frames               = self.paras.get('n_frames',  25)
        self.use_batch_norm         = self.paras.get('batch_norm',  False)
        self.joint_n_conv_layers    = self.paras.get('joint_n_conv_layers',  4)
        self.joint_n_conv_channels  = self.paras.get('joint_n_conv_channels',  16)
        self.joint_conv_size        = self.paras.get('joint_conv_size',  3)
        self.joint_n_output_channels= self.paras.get('joint_n_output_channels',  self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        
        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size  = 1
        
        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)
        joint_bnFunc_conv_inner = nn.BatchNorm2d(self.joint_n_conv_channels) if useBN else nn.Identity()
        joint_bnFunc_conv_output = nn.BatchNorm2d(self.joint_n_output_channels) if useBN else nn.Identity()
        
        joint_convs = [nn.Conv2d(self.n_input_channels, self.joint_n_conv_channels, self.joint_conv_size, 
                           stride=joint_stride_size, padding=joint_padding_size), actiFunc, joint_bnFunc_conv_inner,
                       nn.MaxPool2d((1,2), stride = (1,2))]
        for innerLayerIdx in range(self.joint_n_conv_layers-2):
            joint_convs += [nn.Conv2d(self.joint_n_conv_channels, self.joint_n_conv_channels, self.joint_conv_size, 
                                stride = joint_stride_size, padding = joint_padding_size), 
                            actiFunc, joint_bnFunc_conv_inner,
                            nn.MaxPool2d((1,2), stride = (1,2))]            
        joint_convs += [nn.Conv2d(self.joint_n_conv_channels, self.joint_n_output_channels, self.joint_conv_size, 
                            stride = joint_stride_size, padding = joint_padding_size), 
                  actiFunc, joint_bnFunc_conv_output]
        
        self.joint_conv_layers = nn.ModuleList(joint_convs)
        
        
        # convs += [nn.MaxPool2d((1,2), stride = (1,2))]
        # convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), bnFunc, actiFunc]
        
        # Regression Conv layers
        self.reg_conv_size         = self.paras.get('reg_conv_size', self.joint_conv_size)
        self.reg_n_conv_layers     = self.paras.get('reg_n_conv_layers', self.joint_n_conv_layers)
        # self.reg_n_conv_layers     = 1
        self.reg_n_conv_channels   = self.paras.get('reg_n_conv_channels', self.joint_n_conv_channels)
        self.reg_n_output_channels = self.paras.get('reg_n_output_channels',  self.reg_n_conv_channels)
        self.reg_n_linear_layers   = self.paras.get('reg_n_linear_layers', 3)
        reg_bnFunc_conv_inner = nn.BatchNorm2d(self.reg_n_conv_channels) if useBN else nn.Identity()
        reg_bnFunc_conv_output = nn.BatchNorm2d(self.reg_n_output_channels) if useBN else nn.Identity()
        
        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1
        
        if self.reg_n_conv_layers >= 2:
            reg_convs = [nn.Conv2d(self.joint_n_output_channels, self.reg_n_conv_channels, self.reg_conv_size, 
                            stride=reg_conv_stride, padding=reg_padding_size), 
                        actiFunc, reg_bnFunc_conv_inner]
            for innerLayerIdx in range(self.reg_n_conv_layers-2):
                reg_convs += [nn.Conv2d(self.reg_n_conv_channels, self.reg_n_conv_channels, self.reg_conv_size, 
                                    stride = reg_conv_stride, padding = reg_padding_size), 
                            actiFunc, reg_bnFunc_conv_inner]
            reg_convs += [nn.Conv2d(self.reg_n_conv_channels, self.reg_n_output_channels, self.reg_conv_size, 
                                stride=reg_conv_stride, padding=reg_padding_size), 
                            actiFunc, reg_bnFunc_conv_output]
        elif self.reg_n_conv_layers == 1:
            reg_convs = [nn.Conv2d(self.joint_n_output_channels, self.reg_n_output_channels, self.reg_conv_size, 
                            stride=reg_conv_stride, padding=reg_padding_size), 
                        actiFunc, reg_bnFunc_conv_output]
        elif self.reg_n_conv_layers == 0:
            reg_convs = [nn.Identity()]
            self.reg_n_output_channels = self.joint_n_output_channels
        
        
        # Regression Linear Layers
        reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**self.n_joint_pooling))        
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        # if self.n_joint_pooling > 0:
        #     reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**(self.n_joint_pooling-1)))            
        #     reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**(self.n_joint_pooling-1)))            
        # else:
        #     reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels
        #     reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels

        reg_bnFunc_linear_inner = nn.BatchNorm1d(reg_linear_inner_feature_dim) if useBN else nn.Identity()
        # bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        reg_linears = [nn.Flatten(), 
                       nn.Linear(reg_linear_input_feature_dim, reg_linear_inner_feature_dim),
                       actiFunc, reg_bnFunc_linear_inner]
        for inner_layer_idx in range(self.reg_n_linear_layers - 2):
            reg_linears += [nn.Linear(reg_linear_inner_feature_dim, reg_linear_inner_feature_dim), actiFunc, reg_bnFunc_linear_inner]
        reg_linears += [nn.Linear(reg_linear_inner_feature_dim, self.n_sectors_out)]
        
        self.reg_layers = nn.ModuleList(reg_convs + reg_linears)
        
        # Classification Conv Layers
        self.n_classes             = self.paras.get('n_classes', 1)
        self.cls_conv_size         = self.paras.get('cls_conv_size', self.joint_conv_size)
        self.cls_n_conv_layers     = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        # self.cls_n_conv_layers     = 0
        self.cls_n_conv_channels   = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_n_output_channels = self.paras.get('cls_n_output_channels',  self.cls_n_conv_channels)
        self.cls_n_linear_layers   = self.paras.get('cls_n_linear_layers', 3)
        cls_bnFunc_conv_inner = nn.BatchNorm2d(self.cls_n_conv_channels) if useBN else nn.Identity()
        cls_bnFunc_conv_output = nn.BatchNorm2d(self.cls_n_output_channels) if useBN else nn.Identity()
        
        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1
        
        if self.cls_n_conv_layers >= 2:
            cls_convs = [nn.Conv2d(self.joint_n_output_channels, self.cls_n_conv_channels, self.cls_conv_size, 
                            stride=cls_conv_stride, padding=cls_padding_size), 
                        actiFunc, cls_bnFunc_conv_inner]
            for innerLayerIdx in range(self.cls_n_conv_layers-2):
                cls_convs += [nn.Conv2d(self.cls_n_conv_channels, self.cls_n_conv_channels, self.cls_conv_size, 
                                    stride = cls_conv_stride, padding = cls_padding_size), 
                            actiFunc, cls_bnFunc_conv_inner]
            cls_convs += [nn.Conv2d(self.cls_n_conv_channels, self.cls_n_output_channels, self.cls_conv_size, 
                                stride=cls_conv_stride, padding=cls_padding_size), 
                            actiFunc, cls_bnFunc_conv_output]
        elif self.cls_n_conv_layers == 1:
            cls_convs = [nn.Conv2d(self.joint_n_output_channels, self.cls_n_output_channels, self.cls_conv_size, 
                            stride=cls_conv_stride, padding=cls_padding_size), 
                        actiFunc, cls_bnFunc_conv_output]
        elif self.cls_n_conv_layers == 0:
            cls_convs = [nn.Identity()]
            self.cls_n_output_channels = self.joint_n_output_channels
        
        # Classification Linear Layers
        cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (1*(4**self.n_joint_pooling))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim //2
        # reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (1*(4**self.n_joint_pooling//2))        
        # reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        # if self.n_joint_pooling > 0:
        #     cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (1*(4**2))
        #     cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**2))
        # else:
        #     cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels
        #     cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels

        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        cls_bnFunc_linear_inner = nn.BatchNorm1d(cls_linear_inner_feature_dim) if useBN else nn.Identity()
        # bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        cls_linears = [nn.Flatten(), 
                       nn.Linear(cls_linear_input_feature_dim, cls_linear_inner_feature_dim),
                       actiFunc, cls_bnFunc_linear_inner]
        for inner_layer_idx in range(self.cls_n_linear_layers - 2):
            cls_linears += [nn.Linear(cls_linear_inner_feature_dim, cls_linear_inner_feature_dim), actiFunc, cls_bnFunc_linear_inner]
        cls_linears += [nn.Linear(cls_linear_inner_feature_dim, self.n_classes*self.n_sectors_out)]        
        
        self.cls_layers = nn.ModuleList(cls_convs + cls_linears)                
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type('strainmat', input_types)
    
    def set_output_types(self, output_types):
        self.TOS_type = get_data_type('TOS', output_types)
        self.classify_label_type = get_data_type('sector_dist_map', output_types)
    
    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        # x = self.net(x)        
        # raise ValueError('STOP!')
        # print('init x: ', x.shape)
        for layer in self.joint_conv_layers:
            x = layer(x)
        
        # for layer in self.joint_linear_layers:
        #     x = layer(x)
        # print('After conv layers: ', x.shape)        
        x_cls = x
        for layer in self.cls_layers:
            # print(x_cls.shape)
            # print(layer)
            x_cls = layer(x_cls)
        x_cls = x_cls.view(x_cls.shape[0], self.n_classes, -1)
        # print(x_cls.shape)
        # x_cls = torch.reshape(x_cls, (-1,2,-1))
        
        x_reg = x
        for layer in self.reg_layers:
            # print(x_reg.shape)
            # print(layer)            
            x_reg = layer(x_reg)        
        # x = nn.LeakyReLU(inplace = True)(x-17)+17
        # print(x.shape)
        x_reg = nn.LeakyReLU(inplace = True)(x_reg-17)+17
        # print('x_reg: ', x_reg.shape)
        # print('FWD!')
        return {self.TOS_type: x_reg, self.classify_label_type: x_cls}

class NetStrainMat2LateActiCls(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2LateActiCls, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.n_input_channels   = self.paras.get('n_input_channels', 1)
        self.n_sectors_in       = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out      = self.paras.get('n_sectors_out', 18)
        self.n_frames           = self.paras.get('n_frames',  25)
        self.n_conv_layers      = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels    = self.paras.get('n_conv_channels',  16)
        self.conv_size          = self.paras.get('conv_size',  3)
        self.conv_stride        = self.paras.get('conv_stride', 1)
        
        useBN = False
        # actiFunc = nn.Sigmoid()
        actiFunc = nn.ReLU(True)
        # actiFunc = nn.LeakyReLU(True)
        bnFuncConv1 = nn.BatchNorm2d(self.n_conv_channels) if useBN else nn.Identity()
        bnFuncConv2 = nn.BatchNorm2d(1) if useBN else nn.Identity()
        convs = [nn.Conv2d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=self.conv_stride, padding=self.conv_size//2), actiFunc, bnFuncConv1]
        for innerLayerIdx in range(self.n_conv_layers-2):
            convs += [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=self.conv_stride, padding=self.conv_size//2), actiFunc, bnFuncConv1]
        convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=self.conv_stride, padding=self.conv_size//2), actiFunc, bnFuncConv2]
        
        
        # linear = [nn.Flatten(), nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_out)]
        
        # linear = [nn.Flatten(), \
        #           nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_in*self.n_frames//2),\
        #           bnFunc, actiFunc, \
        #           nn.Linear(self.n_sectors_in*self.n_frames//2, self.n_sectors_out)]
        bnFuncLinear1 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//2) if useBN else nn.Identity()
        bnFuncLinear2 = nn.BatchNorm1d(self.n_sectors_in*self.n_frames//4) if useBN else nn.Identity()
        linear = [nn.Flatten(), \
                  nn.Linear(self.n_sectors_in*self.n_frames, self.n_sectors_in*self.n_frames//2),\
                  actiFunc,bnFuncLinear1,  \
                  nn.Linear(self.n_sectors_in*self.n_frames//2, self.n_sectors_in*self.n_frames//4),
                  actiFunc,bnFuncLinear2,  \
                  nn.Linear(self.n_sectors_in*self.n_frames//4, self.n_sectors_out)]
        self.layers = nn.ModuleList(convs+linear)
        
    @autocast()
    def forward(self, input_dict):
        # with autocast():
        x = input_dict[self.input_type]#.cuda()
        # print(self.input_type, x.dtype)
        # x = self.net(x)        
        for layer in self.layers:
            # print(x.shape)
            # print(layer)
            # print(layer.weight.type())
            x = layer(x)        
        x = nn.LeakyReLU(inplace = True)(x-17)+17
        # return x
        return {self.output_types: x}
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.input_type = input_types
        elif type(input_types) is list:
            self.input_type = get_data_type('strainmat', input_types)
    
    def set_output_types(self, output_types):
        if type(output_types) is str:
            self.output_types = output_types
        elif type(output_types) is list:
            self.output_types = get_data_type('TOS', output_types)