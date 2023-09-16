# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:49:29 2021

@author: Jerry Xing
"""

import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast
from utils.data import get_data_type_by_category
from icecream import ic

def get_last_conv_channels_num(layers: list):
    last_conv_layer = [layer for layer in layers if type(layer) is nn.Conv2d][-1]
    return last_conv_layer.out_channels
    

def get_pooling_layer_num(layers: list):
    pooling_layers = len([layer for layer in layers if type(layer) is nn.MaxPool2d])
    return len(pooling_layers)

def conv2d_net(input_shape: list or tuple or int, 
             n_conv_layers=3,
             n_output_channels:int or None=None,
             conv_size=3,
             stride_size=1, 
             padding_size=1, 
             pooling_method=None, 
             pooling_kernel_size=2,
             pooling_layers_max_num=None,
             activation_func=torch.nn.ReLU(), 
             batchnorm=False, 
             batchnorm_at_end=False, activation_at_end=False):
    # https://stackoverflow.com/questions/42015156/the-order-of-pooling-and-normalization-layer-in-convnet
    # pooling -> batch normalizaion
    if n_conv_layers <=0:
        return []
    
    # Input shape could be:
        # 1) (C, H, W)
        # 2) (C, None, None)
        # 3) C
    if type(input_shape) in [list, tuple]:
        if len(input_shape) == 4:
            _, input_channel_num, input_H, input_W = input_shape
        elif len(input_shape) == 3:
            input_channel_num, input_H, input_W = input_shape
        elif len(input_shape) == 2:
            raise ValueError('Unsupported intput shape: ', input_shape)
        elif len(input_shape) == 1:
            input_channel_num = input_shape[0]
            input_H, input_W = None, None
    
    actual_pooling_layer_num = 0
    # Determine pooling settings
    if pooling_method is not None:
        if pooling_layers_max_num is None:
            # if contain pooling layers and pooling number is not specified, set number equals to number of conv layers
            pooling_layers_max_num = n_conv_layers
    else:
        pooling_layers_max_num = 0
    
    # Determine channel number of each layer according to specified output channel number or pooling method
    if n_output_channels is not None:
        # if output channel num is specified, linear interpolated inner layer channel numbers
        conv_input_channels = np.linspace(input_channel_num, n_output_channels, n_conv_layers + 1).astype(int)#[:-1]
    else:        
        if pooling_method is None or n_output_channels == 0:
            # if output channel num is not specified and don't use pooling, keep channel number unchanged
            conv_input_channels = [input_channel_num] * (n_conv_layers + 1) # +1 to include the output of last layer
        else:
            # if output channel num is not specified and use pooling, the channel numbers are decided by doubling
            conv_input_channels = []
            for layer_idx in range(pooling_layers_max_num):
                conv_input_channels.append(input_channel_num * 2**layer_idx)
            conv_input_channels += [input_channel_num * 2**(pooling_layers_max_num-1)] * (n_conv_layers - pooling_layers_max_num + 1)        
    
    # Setting layers
    # ic(conv_input_channels)
    layers = []
    for layer_idx in range(n_conv_layers):
        # Conv
        
        layers.append(
                nn.Conv2d(in_channels=conv_input_channels[layer_idx], 
                          out_channels=conv_input_channels[layer_idx + 1], 
                          kernel_size=conv_size,
                          stride=stride_size, 
                          padding=padding_size)
                )
    
        # Pooling
        if (layer_idx + 1) <= pooling_layers_max_num:
            # if add pooling layer            
            layers.append(nn.MaxPool2d(pooling_kernel_size, stride=2))
            actual_pooling_layer_num += 1
        
        # Batchnorm
        if batchnorm:
            if layer_idx != n_conv_layers - 1 or batchnorm_at_end:
                # if it's not the last layer or it's forced to be added after the last layer
                layers.append(nn.BatchNorm2d(conv_input_channels[layer_idx + 1]))
        
        # Activation
        if layer_idx != n_conv_layers - 1 or activation_at_end:
            layers.append(activation_func)
            
    if input_H is not None and input_W is not None:
        output_H = input_H // (2**actual_pooling_layer_num)
        output_W = input_W // (2**actual_pooling_layer_num)
    else:
        output_H, output_W = None, None
    
    # ic(layers)
    # ic([type(layer) for layer in layers if type(layers) is nn.Conv2d])
    output_channel_num = [layer for layer in layers if type(layer) == nn.Conv2d][-1].out_channels
    return {'layers': layers,
            'output_shape': (output_channel_num, output_H, output_W)}
    # return layers

# def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc,
#         useBN=False):
#     if n_linear_layers > 1:
#         linear_layers = [nn.Flatten(),
#                          nn.Linear(linear_input_feature_dim, linear_inner_feature_dim),
#                          actiFunc]
#         if useBN:
#             linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
    
#         for inner_layer_idx in range(n_linear_layers - 2):
#             linear_layers += [nn.Linear(linear_inner_feature_dim, linear_inner_feature_dim), actiFunc]
#             if useBN:
#                 linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
    
#         linear_layers += [nn.Linear(linear_inner_feature_dim, linear_output_feature_dim)]
#     else:
#         linear_layers = [nn.Flatten(),
#                          nn.Linear(linear_input_feature_dim, linear_output_feature_dim),
#                          actiFunc]
#         if useBN:
#             linear_layers += [nn.BatchNorm1d(linear_output_feature_dim)]
#     return linear_layers

def fcn(n_linear_layers, 
        linear_input_feature_dim, 
        linear_inner_feature_dim, 
        linear_output_feature_dim, 
        activation_func,
        batchnorm=False, 
        simple_last_layer=True):
    
    if n_linear_layers <=0:
        return []
    
    if n_linear_layers > 1:
        layers = [nn.Flatten()]
        
        
        if linear_inner_feature_dim is None:
            layer_input_dims = np.linspace(linear_input_feature_dim, linear_output_feature_dim, n_linear_layers + 1).astype(int)
        elif type(linear_inner_feature_dim) is int:
            layer_input_dims = [linear_input_feature_dim] + [linear_inner_feature_dim]*(n_linear_layers-2+1) + [linear_output_feature_dim]
        
        ic(layer_input_dims)                                
        for layer_idx in range(n_linear_layers):
            curr_layer_input_dim = layer_input_dims[layer_idx]
            curr_layer_output_dim = layer_input_dims[layer_idx + 1]
                        
            layers.append(nn.Linear(curr_layer_input_dim, curr_layer_output_dim))
            
            if layer_idx != n_linear_layers-1 or not simple_last_layer:
                layers.append(activation_func)
                if batchnorm:
                    layers.append(nn.BatchNorm1d(curr_layer_output_dim))               
    else:
        layers = [nn.Flatten(),
                         nn.Linear(linear_input_feature_dim, linear_output_feature_dim),
                         activation_func]
        if batchnorm:
            layers += [nn.BatchNorm1d(linear_output_feature_dim)]
    # return layers
    return {'layers': layers}

def conv_fc_net(
        conv_intput_shape: list or tuple,
        linear_output_dim:int, 
        conv_layer_num=3,
        conv_output_channel_num:int or None=None,
        conv_kernel_size=3,
        conv_stride_size=1, 
        conv_padding_size=1, 
        conv_pooling_method=None, 
        conv_pooling_kernel_size=2,
        conv_pooling_layers_max_num=None,
        activation_func=torch.nn.ReLU(), 
        batchnorm=False, 
        linear_layer_num=3,        
        linear_inner_dim = None,
        simple_last_linear_layer=True
        ):
    
    
    conv_net = conv2d_net(
        input_shape = conv_intput_shape, 
        n_conv_layers = conv_layer_num, 
        n_output_channels = conv_output_channel_num,
        conv_size = conv_kernel_size,
        stride_size = conv_stride_size, 
        padding_size = conv_padding_size,                  
        pooling_method = conv_pooling_method, 
        pooling_kernel_size = conv_pooling_kernel_size,
        pooling_layers_max_num = conv_pooling_layers_max_num,
        activation_func = activation_func, 
        batchnorm = batchnorm,
        batchnorm_at_end = True, 
        activation_at_end = True)
    
    linear_net = fcn(
        n_linear_layers = linear_layer_num, 
        linear_input_feature_dim = np.prod(conv_net['output_shape']), 
        linear_inner_feature_dim = linear_inner_dim,
        linear_output_feature_dim = linear_output_dim, 
        activation_func = activation_func, 
        batchnorm = batchnorm,
        simple_last_layer = True)
    
    # return conv_net['layers'] + linear_net['layers']
    return {'layers': conv_net['layers'] + linear_net['layers'], 'output_shape': linear_output_dim}

class View(nn.Module):
    """ Custom reshape layer """
    def __init__(self, target_shape):
        # https://github.com/pytorch/vision/issues/720
        super().__init__()   
        # self.shift = shift
        self.target_shape = target_shape

    @autocast()
    def forward(self, x):        
        # x = x.reshape(x.shape[0], self.n_classes, -1)
        # return x.view(self.target_shape)
        return x.reshape(self.target_shape)

class shift_Leaky_ReLU(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, shift):
        super().__init__()   
        self.shift = shift
    
    @autocast()
    def forward(self, x):        
        return nn.LeakyReLU(inplace=True)(x - self.shift) + self.shift

class NetStrainMat2ClsReg(nn.Module):
    def __init__(self, config={}):
        super(NetStrainMat2ClsReg, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.input_channel_num = self.paras.get('input_channel_num', 1)        
        self.input_sector_num = self.paras.get('input_sector_num', 18)
        # self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.input_frame_num = self.paras.get('input_frame_num', 25)        
        
        self.joint_init_conv_channel_num = self.paras.get('joint_init_conv_channel_num', 16)
        self.joint_conv_layer_num = self.paras.get('joint_conv_layer_num', 4)
        # self.joint_n_conv_channels = self.paras.get('joint_n_conv_channels', 16)
        self.joint_conv_channel_num = self.paras.get('joint_conv_channel_num', 16)
        self.joint_conv_size = self.paras.get('joint_conv_size', 3)
        # self.joint_conv_output_channel_num = self.paras.get('joint_conv_output_channel_num', self.joint_n_conv_channels)
        self.joint_pooling_layer_num_max = self.paras.get('joint_pooling_layer_num_max', None)
        self.conv_output_channel_num = self.paras.get('conv_output_channel_num', None)
        
        self.pooling_method = self.paras.get('pooling_method', 'maxpooling')
        self.activation_func = self.paras.get('activation_func', 'ReLU')
        self.use_batch_norm = self.paras.get('batch_norm', True)                
        self.classes_num = self.paras.get('classes_num', 2)

        joint_conv_padding_size = self.joint_conv_size // 2
        joint_conv_stride_size = 1
        
        if self.activation_func.lower() == 'relu':
            actiFunc = nn.ReLU()
        elif self.activation_func.lower() == 'leaky_relu':
            actiFunc = nn.LeakyReLU()
        elif self.activation_func.lower() == 'sigmoid':
            actiFunc = nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation function: ', self.activation_func)

        # self.joint_conv_layers = nn.ModuleList(
        #     conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size,
        #              joint_stride_size, joint_padding_size, self.joint_n_output_channels,
        #              self.n_joint_pooling, actiFunc, useBN=self.use_batch_norm))
        
        joint_init_channel_conv_net = conv2d_net(
            input_shape = (self.input_channel_num, self.input_sector_num, self.input_frame_num), 
            n_conv_layers = 1, 
            n_output_channels = self.joint_init_conv_channel_num,
            conv_size = self.joint_conv_size,
            stride_size = joint_conv_stride_size, 
            padding_size = joint_conv_padding_size,                  
            pooling_method = self.pooling_method, 
            pooling_kernel_size = 2,
            pooling_layers_max_num = 1,
            actiFunc = actiFunc, 
            batchnorm = self.use_batch_norm, 
            batchnorm_at_end = True, 
            activation_at_end = True)
        
        
        joint_contracting_path_conv_net = conv2d_net(
            n_input_channels = joint_init_channel_conv_net['output_shape'], 
            n_conv_layers = self.n_conv_layers - 1, 
            n_output_channels = self.conv_output_channel_num,
            conv_size = self.joint_conv_size,
            stride_size = joint_conv_stride_size, 
            padding_size = joint_conv_padding_size,                  
            pooling_method = self.pooling_method, 
            pooling_kernel_size = 2,
            pooling_layers_max_num = self.joint_pooling_layer_num_max - 1,
            actiFunc = actiFunc, 
            batchnorm = self.use_batch_norm, 
            batchnorm_at_end = True, 
            activation_at_end = True)
        
        self.joint_layers = nn.ModuleList(joint_init_channel_conv_net['layers'] + joint_contracting_path_conv_net['layers'])
        # joint_last_conv_layer = get_last_conv_channels_num(joint_conv_layers)
        # joint_pooling_layer_num = get_pooling_layer_num(joint_last_conv_layer)    
        
        # Regression Conv layers
        self.reg_conv_kernel_size = self.paras.get('reg_conv_kernel_size', 3)
        self.reg_conv_layer_num = self.paras.get('reg_conv_layer_num', 3)
        # self.reg_conv_channel_num = self.paras.get('reg_conv_channel_num', None)
        self.reg_conv_output_channel_num = self.paras.get('reg_conv_output_channel_num', None)
        self.reg_pooling_layer_num_max = self.paras.get('reg_pooling_layer_num_max', None)
        self.reg_linear_layer_num = self.paras.get('reg_linear_layer_num', 3)
        self.reg_linear_layer_inner_dim = self.get('reg_linear_inner_layer_dim', None)
        self.reg_output_dim = self.paras.get('reg_output_dim', 18)

        reg_padding_size = self.reg_conv_size // 2
        reg_conv_stride = 1
        
        reg_net = conv_fc_net(
            conv_intput_shape = joint_contracting_path_conv_net['output_shape'], 
            linear_output_dim = self.reg_output_dim, 
            conv_layer_num = self.reg_conv_layer_num,
            conv_output_channel_num = self.reg_conv_output_channel_num,
            conv_kernel_size = self.reg_conv_kernel_size,
            conv_stride_size = reg_conv_stride, 
            conv_padding_size = reg_padding_size, 
            conv_pooling_method = self.pooling_method, 
            conv_pooling_kernel_size = 2,
            conv_pooling_layers_max_num = self.reg_pooling_layer_num_max,
            actiFunc = actiFunc, 
            batchnorm = self.use_batch_norm, 
            linear_layer_num = self.reg_linear_layer_num,
            linear_inner_dim = self.reg_linear_layer_inner_dim,
            simple_last_linear_layer = True)
        
        # reg_conv_layers = conv_net(
        #     n_input_channels = self.n_init_conv_channels, 
        #     n_conv_layers = self.reg_conv_layer_num, 
        #     n_output_channels = self.conv_output_channel_num,
        #     conv_size = self.joint_conv_size,
        #     stride_size = reg_conv_stride, 
        #     padding_size = reg_padding_size,                  
        #     pooling_method = self.pooling_method, 
        #     pooling_kernel_size = 2,
        #     pooling_layers_max_num = self.reg_pooling_layer_num_max,
        #     actiFunc = actiFunc, 
        #     batchnorm = self.use_batch_norm,
        #     batchnorm_at_end = True, 
        #     activation_at_end = True)
        
        # reg_last_conv_layer = get_last_conv_channels_num(reg_conv_layers)
        # reg_pooling_layer_num = get_pooling_layer_num(reg_conv_layers)    

        # # Regression Linear Layers
        # # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**2))
        # reg_linear_input_feature_dim = self.input_sector_num * self.input_frame_num * reg_last_conv_layer.out_channels // (
        #             1 * (4 ** (joint_pooling_layer_num + reg_pooling_layer_num)))
        # # reg_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.reg_n_output_channels // (2*(4**self.n_joint_pooling/2))
        # # reg_linear_inner_feature_dim = reg_linear_input_feature_dim // 2
        # # reg_linear_output_feature_dim = self.reg_n_dim_out

        # # reg_linears = fcn(self.reg_n_linear_layers, reg_linear_input_feature_dim, reg_linear_inner_feature_dim,
        # #                   reg_linear_output_feature_dim,
        # #                   actiFunc, useBN=self.use_batch_norm)
        
        # reg_linear_layers = fcn(
        #     n_linear_layers = self.n_linear_layers, 
        #     linear_input_feature_dim = reg_linear_input_feature_dim, 
        #     linear_inner_feature_dim = self.reg_linear_inner_feature_dim, 
        #     linear_output_feature_dim = self.cls_n_dim_out, 
        #     actiFunc = actiFunc, 
        #     useBN=self.use_batch_norm)
        
        
        # print(reg_convs, reg_linears)
        # self.reg_layers = nn.ModuleList(reg_convs + reg_linears)
        # self.reg_layers = reg_conv_layers + reg_linear_layers
        self.reg_layers = nn.ModuleList(reg_net['layers'])
        
        

        # Classification Conv Layers        
        self.cls_conv_kernel_size = self.paras.get('cls_conv_kernel_size', self.joint_conv_size)
        self.cls_conv_layer_num = self.paras.get('cls_n_conv_layers', self.joint_n_conv_layers)
        # self.cls_n_conv_channels = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_conv_output_channel_num = self.paras.get('cls_conv_output_channel_num', self.cls_n_conv_channels)
        self.cls_pooling_layer_num_max = self.paras.get('cls_pooling_layer_num_max', None)
        self.cls_linear_layer_num = self.paras.get('cls_linear_layer_num', 3)
        self.cls_linear_layer_inner_dim = self.get('cls_linear_inner_layer_dim', None)
        self.cls_output_dim = self.paras.get('cls_output_dim', 18)

        cls_padding_size = self.cls_conv_size // 2
        cls_conv_stride = 1

        # cls_convs = conv_net(self.joint_n_output_channels, self.cls_n_conv_layers, self.cls_n_conv_channels,
        #                      self.cls_conv_size,
        #                      cls_conv_stride, cls_padding_size, self.cls_n_output_channels,
        #                      0, actiFunc, useBN=self.use_batch_norm)
        
        cls_net = conv_fc_net(
            conv_intput_shape = joint_contracting_path_conv_net['output_shape'], 
            linear_output_dim = self.cls_output_dim, 
            conv_layer_num = self.cls_conv_layer_num,
            conv_output_channel_num = self.cls_conv_output_channel_num,
            conv_kernel_size = self.cls_conv_kernel_size,
            conv_stride_size = cls_conv_stride, 
            conv_padding_size = cls_padding_size, 
            conv_pooling_method = self.pooling_method, 
            conv_pooling_kernel_size = 2,
            conv_pooling_layers_max_num = self.cls_pooling_layer_num_max,
            actiFunc = actiFunc, 
            batchnorm = self.use_batch_norm, 
            linear_layer_num = self.cls_linear_layer_num,
            linear_inner_dim = self.cls_linear_layer_inner_dim,
            simple_last_linear_layer = True)

        # Classification Linear Layers
        # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels//(2*(4**2))
        # cls_linear_input_feature_dim = self.n_sectors_in * self.n_frames * self.cls_n_output_channels // (
        #             1 * (4 ** self.n_joint_pooling))
        # # cls_linear_inner_feature_dim = self.n_sectors_in*self.n_frames*self.cls_n_output_channels // (2*(4**self.n_joint_pooling/2))
        # cls_linear_inner_feature_dim = cls_linear_input_feature_dim // 2

        # cls_linears = fcn(self.cls_n_linear_layers, cls_linear_input_feature_dim, cls_linear_inner_feature_dim,
        #                   self.cls_n_dim_out,
        #                   actiFunc, useBN=self.use_batch_norm)

        # self.cls_layers = nn.ModuleList(cls_convs + cls_linears)
        self.cls_layers = nn.ModuleList(cls_net['layers'])

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
        for layer in self.joint_layers:
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
    
class NetStrainMat2Cls(nn.Module):
    def __init__(self, config={}):
        # super(NetStrainMat2Cls, self).__init__()
        super().__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.input_channel_num = self.paras.get('input_channel_num', 1)
        self.init_conv_channel_num = self.paras.get('init_conv_channel_num', 16)
        self.input_sector_num = self.paras.get('input_sector_num', 18)
        self.input_frame_num = self.paras.get('input_frame_num', 25)
        self.conv_layer_num = self.paras.get('conv_layer_num', 3)        
        self.conv_output_channel_num = self.paras.get('conv_output_channel_num', None)
        self.pooling_method = self.paras.get('pooling_method', 'maxpooling')
        self.pooling_layer_num_max = self.paras.get('pooling_layer_num_max', None)
        self.conv_kernel_size = self.paras.get('conv_kernel_size', 3)
        self.conv_stride = self.paras.get('conv_stride', 1)
        self.linear_layer_num = self.paras.get('linear_layer_num', 3)
        self.cls_output_dim = self.paras.get('cls_output_dim', 256)
        self.classes_num = self.paras.get('classes_num', 2)
        self.activation_func = self.paras.get('activation_func', 'ReLU')
        self.force_onehot = self.paras.get('force_onehot', False)
        self.class_normlize_layer = self.paras.get('class_normlize_layer', 'softmax')
        

        # useBN = False
        self.use_batch_norm = self.paras.get('batch_norm', True)
        if self.activation_func.lower() == 'relu':
            # actiFunc = nn.ReLU(True)
            activation_func = nn.ReLU()
        elif self.activation_func.lower() == 'leaky_relu':
            # actiFunc = nn.LeakyReLU(True)
            activation_func = nn.LeakyReLU()
        elif self.activation_func.lower() == 'sigmoid':
            activation_func = nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation function: ', self.activation_func)
        
        conv_padding_size = self.conv_kernel_size // 2
        
        # Add one initial conv layer to expand the feature channels
        # This layer should keep height and width of input data
        # self.n_init_conv_channels = self.paras.get('n_init_conv_channels', 16)
        # conv_layers = []
        # conv_layers
        init_net = conv2d_net(
            input_shape = (self.input_channel_num, self.input_sector_num, self.input_frame_num), 
            n_conv_layers = 1, 
            n_output_channels = self.init_conv_channel_num,
            conv_size = self.conv_kernel_size,
            stride_size = self.conv_stride, 
            padding_size = conv_padding_size,                  
            pooling_method = self.pooling_method, 
            pooling_kernel_size = 2,
            pooling_layers_max_num = None,
            activation_func = activation_func, 
            batchnorm = self.use_batch_norm, 
            batchnorm_at_end = True, 
            activation_at_end = True)
        
        # convs += conv2d_net(
        #     n_input_channels = self.n_init_conv_channels, 
        #     n_conv_layers = self.n_conv_layers - 1, 
        #     n_output_channels = self.n_conv_output_channels,
        #     conv_size = self.conv_size,
        #     stride_size = self.conv_stride, 
        #     padding_size = conv_padding_size,                  
        #     pooling_method = self.pooling_method, 
        #     pooling_kernel_size = 2,
        #     pooling_layers_max_num = None,
        #     actiFunc = actiFunc, 
        #     batchnorm = self.use_batch_norm, 
        #     batchnorm_at_end = True, 
        #     activation_at_end = True)
        
        # # convs = conv_net(self.n_input_channels, self.n_conv_layers, self.n_conv_channels,
        # #                      self.conv_size,
        # #                      self.conv_stride, conv_padding_size, self.n_conv_output_channels,
        # #                      0, actiFunc, useBN=self.use_batch_norm)
        # n_maxpooling_layers = len([layer for layer in convs if type(layer) is nn.MaxPool2d])
        # n_conv_output_channels = [layer for layer in convs if type(layer) is nn.Conv2d][-1].out_channels
        # # cl.out_channels
        
        # # def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc, useBN=False)
        # linear_input_dim = \
        #     self.n_sectors_in//(2**n_maxpooling_layers) * \
        #     self.n_frames//(2**n_maxpooling_layers) * \
        #     n_conv_output_channels
        # # linear = fcn(self.n_linear_layers, linear_input_dim, linear_input_dim // 2, self.cls_n_dim_out, actiFunc,
        # linear = fcn(self.n_linear_layers, linear_input_dim, None, self.cls_n_dim_out, actiFunc, useBN=self.use_batch_norm)
        # self.layers = nn.ModuleList(convs + linear)
        
        following_net = conv_fc_net(
            conv_intput_shape = init_net['output_shape'],
            linear_output_dim = self.cls_output_dim, 
            conv_layer_num = self.conv_layer_num - 1,
            conv_output_channel_num = self.conv_output_channel_num,
            conv_kernel_size = self.conv_kernel_size,
            conv_stride_size = self.conv_stride, 
            conv_padding_size = 1, 
            conv_pooling_method = self.pooling_method, 
            conv_pooling_kernel_size = 2,
            conv_pooling_layers_max_num = self.pooling_layer_num_max,
            activation_func = activation_func, 
            batchnorm = False, 
            linear_layer_num = self.linear_layer_num,
            linear_inner_dim = None,
            simple_last_linear_layer = True
            )#['layers']
        
        layers = init_net['layers'] + following_net['layers']
        
        if self.force_onehot or self.classes_num > 2:
            # Reshape data to (batch_size, class_num, sector_num)
            layers.append(View((-1, self.classes_num, self.cls_output_dim // self.classes_num)))            
            if self.class_normlize_layer.lower() == 'softmax':
                layers.append(nn.Softmax(dim = -2))
            elif self.class_normlize_layer.lower() == 'log softmax':
                layers.append(nn.LogSoftmax(dim=-2))
        else:
            layers.append(nn.LeakyReLU())
            
        self.layers = nn.ModuleList(layers)

    @autocast()
    def forward(self, input_data: dict or torch.Tensor):
        # if type(input_data) is dict:
        #     x = input_data[self.input_type]  # .cuda()
        # else:
        #     x = input_data
        # print('HA!')
        x = input_data[self.input_type]  # .cuda()
        # print('INPUT')
        # print('    ', x.shape)
        # print('    ', torch.isnan(x).any())
        for layer in self.layers:
            # print(layer)
            # x_before_layer = x
            x = layer(x)
            # print('    ', x.shape)
            # if torch.isnan(x).any():
            #     print(x_before_layer[0])
            #     print(x[0])
            # print(torch.isnan(x).any())
            #     print('    ', torch.isnan(x).any())
                # break
            # return
                
        # print(x[0,0,:10])
        
        # print('-----------------')
        # print('-----------------')
        
        # print(self.n_classes)
        # print(x.shape)
        # print(self.force_onehot)
        # if self.force_onehot or self.n_classes > 2:
        #     # print(x.shape[0], self.n_classes, -1)
        #     # x = x.reshape(x.shape[0], self.n_classes, -1)
        #     x = x.reshape(-1, self.classes_num, self.cls_output_dim // self.classes_num)
        #     # xls: [N, n_clasees, n_sectors]
        #     x = nn.Softmax(dim=-11)(x)
        # else:
        #     x = nn.LeakyReLU(inplace=True)(x)
        #     # x = nn.ReLU(inplace=True)(x)
            # pass
        # return x
        # print(x.shape)
        
        # if type(input_data) is dict:
        return {self.output_types: x}
        # else:
            # return x

    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.input_type = input_types
        elif type(input_types) is list:
            self.input_type = get_data_type_by_category('strainmat', input_types)
        if self.input_type is None:
            raise ValueError('Type ', input_types, 'not recognized!')

    def set_output_types(self, output_types):
        if type(output_types) is str:
            self.output_types = output_types
        elif type(output_types) is list:
            sector_label_type = get_data_type_by_category('sector_label', output_types)
            if sector_label_type is not None:
                self.output_types = get_data_type_by_category('sector_label', output_types)            
            else:
                self.output_types = get_data_type_by_category('data_label', output_types)            
        
        if self.output_types is None:
            raise ValueError('Type ', output_types, 'not recognized!')


            
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
            actiFunc = nn.ReLU()
        elif self.activation_func.lower() == 'sigmoid':
            actiFunc = nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation function: ', self.activation_func)
        # actiFunc = nn.ReLU(True)
        # actiFunc = nn.LeakyReLU(True)
        
        conv_padding_size = self.conv_size // 2
        convs = conv2d_net(self.n_input_channels, self.n_conv_layers, self.n_conv_channels,
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
                
if __name__ == '__main__':
    test_network = 'NetStrainMat2Cls'
    if test_network == 'NetStrainMat2Cls':
        from torch.cuda.amp import autocast, GradScaler
        
        debug_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        debug_batch_size = 64
        debug_input_types = ['strainMatFullResolutionSVD']
        debug_input_shapes = [(debug_batch_size, 1, 128, 64)]
        debug_input_data = {}
        for debug_input_idx in range(len(debug_input_types)):
            debug_input_data[debug_input_types[debug_input_idx]] = torch.rand(debug_input_shapes[debug_input_idx]).to(debug_device)
            
        debug_target_types = ['late_activation_sector_label']
        debug_target_shapes = [(debug_batch_size, 2, 128)]
        debug_target_data = {}
        
        
        for debug_target_idx in range(len(debug_target_types)):
            debug_target_data[debug_target_types[debug_target_idx]] = torch.rand(debug_target_shapes[debug_target_idx]).to(debug_device)
        
        
        # self.paras = config.get('paras', {})
        # self.input_channel_num = self.paras.get('input_channel_num', 1)
        # self.n_init_conv_channels = self.paras.get('n_init_conv_channels', 16)
        # self.input_sector_num = self.paras.get('input_sector_num', 18)
        # # self.n_sectors_out = self.paras.get('n_sectors_out', 18)
        # self.input_frame_num = self.paras.get('input_frame_num', 25)
        # self.conv_layer_num = self.paras.get('conv_layer_num', 4)        
        # self.conv_channel_num = self.paras.get('conv_channel_num', 16)
        # # self.n_conv_output_channels = self.paras.get('n_conv_output_channels', 2)
        # self.conv_output_channel_num = self.paras.get('conv_output_channel_num', None)
        # # self.n_pooling_layers = self.paras.get('n_pooling_layers', 0)
        # self.pooling_method = self.paras.get('pooling_method', 'maxpooling')
        # self.pooling_layer_num_max = self.paras.get('pooling_layer_num_max', None)
        # self.conv_kernel_size = self.paras.get('conv_kernel_size', 3)
        # self.conv_stride = self.paras.get('conv_stride', 1)
        # self.linear_layer_num = self.paras.get('linear_layer_num', 3)
        # self.cls_output_dim = self.paras.get('cls_output_dim', 256)
        # self.classes_num = self.paras.get('classes_num', 2)
        # self.activation_func = self.paras.get('activation_func', 'ReLU')
        # self.force_onehot = self.paras.get('force_onehot', False)

        # # useBN = False
        # self.use_batch_norm = self.paras.get('batch_norm', True)
        debug_config = {
            'paras':{
                'input_channel_num': 1,
                'init_conv_channel_num': 16,
                'input_sector_num': 128,
                'input_frame_num': 64,
                'conv_layer_num': 3,
                'conv_output_channel_num': None,
                'pooling_method': 'maxpooling',
                'linear_layer_num': 3,
                'cls_output_dim': 256,
                'classes_num': 2,
                'force_onehot': True
                }
            }
        debug_network = NetStrainMat2Cls(debug_config).to(debug_device)
        
        debug_network.set_input_types(debug_input_types)
        debug_network.set_output_types(debug_target_types)
        
        debug_optimizer = torch.optim.Adam(debug_network.parameters(), lr=1e-4,
                                              weight_decay=1e-5)
        
        debug_scaler = GradScaler(enabled=True)
        
        debug_loss = 0
        with autocast():
            debug_output_data = debug_network(debug_input_data)
            for debug_data_type_idx in range(len(debug_target_types)):
                # debug_loss += torch.nn.MSELoss()(debug_output_data[debug_target_types[debug_data_type_idx]], debug_target_data[debug_target_types[debug_data_type_idx]])
                debug_loss += nn.BCEWithLogitsLoss()(debug_output_data[debug_target_types[debug_data_type_idx]], debug_target_data[debug_target_types[debug_data_type_idx]])
        
        # loss.backward()                
        # optimizer.step()
        
        debug_scaler.scale(debug_loss).backward()
        debug_scaler.step(debug_optimizer)
        debug_scaler.update()
        
        
        # loss.backward()