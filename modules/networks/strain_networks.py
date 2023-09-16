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

from modules.networks.shared_components import conv2d_net, fcn, conv_fc_net, View, shift_Leaky_ReLU


class NetStrainMat2ClsReg(nn.Module):
    def __init__(self, config={}):
        # super(NetStrainMat2ClsReg, self).__init__()
        super().__init__()
        # Joint conv layers
        self.paras = config.get('paras', {})
        self.input_channel_num = self.paras.get('input_channel_num', 1)        
        self.input_sector_num = self.paras.get('input_sector_num', 18)
        # self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.input_frame_num = self.paras.get('input_frame_num', 25)        
        
        self.joint_init_conv_channel_num = self.paras.get('joint_init_conv_channel_num', 16)        
        # self.joint_n_conv_channels = self.paras.get('joint_n_conv_channels', 16)
        self.joint_conv_channel_num = self.paras.get('joint_conv_channel_num', 16)
        self.joint_conv_kernel_size = self.paras.get('joint_conv_kernel_size', 3)
        # self.joint_conv_output_channel_num = self.paras.get('joint_conv_output_channel_num', self.joint_n_conv_channels)
        
        self.joint_conv_output_channel_num = self.paras.get('joint_conv_output_channel_num', None)
        
        self.pooling_method = self.paras.get('pooling_method', 'maxpooling')
        self.activation_func = self.paras.get('activation_func', 'ReLU')
        self.use_batch_norm = self.paras.get('batch_norm', True)                
        self.classes_num = self.paras.get('classes_num', 2)

        joint_conv_padding_size = self.joint_conv_kernel_size // 2
        joint_conv_stride_size = 1
        
        if self.activation_func.lower() == 'relu':
            activation_func = nn.ReLU()
        elif self.activation_func.lower() == 'leaky_relu':
            activation_func = nn.LeakyReLU()
        elif self.activation_func.lower() == 'sigmoid':
            activation_func = nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation function: ', self.activation_func)
            
            
        # Modify pooling layer num to make sure the data won't be too small
        self.joint_conv_layer_num = self.paras.get('joint_conv_layer_num', 4)
        self.joint_pooling_layer_num_max = self.paras.get('joint_pooling_layer_num_max', None)
        self.reg_conv_layer_num = self.paras.get('reg_conv_layer_num', 3)
        self.reg_pooling_layer_num_max = self.paras.get('reg_pooling_layer_num_max', None)
        self.cls_conv_layer_num = self.paras.get('cls_n_conv_layers', 3)
        self.cls_pooling_layer_num_max = self.paras.get('cls_pooling_layer_num_max', None)
        
        if self.joint_pooling_layer_num_max is None:
            joint_pooling_layer_num_max = self.joint_conv_layer_num
        else:
            joint_pooling_layer_num_max = self.joint_pooling_layer_num_max
        if self.reg_pooling_layer_num_max is None:
            reg_pooling_layer_num_max = self.reg_conv_layer_num
        else:
            reg_pooling_layer_num_max = self.reg_pooling_layer_num_max
        if self.cls_pooling_layer_num_max is None:
            cls_pooling_layer_num_max = self.cls_conv_layer_num
        else:
            cls_pooling_layer_num_max = self.cls_pooling_layer_num_max

        # valid_pooling_num = 0
        for power in range(20):
            if 2**power >= min(self.input_sector_num, self.input_frame_num):
                valid_pooling_num_max = power - 1
                break
        
        if joint_pooling_layer_num_max + reg_pooling_layer_num_max > valid_pooling_num_max:
            self.reg_pooling_layer_num_max = max(0, valid_pooling_num_max - joint_pooling_layer_num_max)
        if joint_pooling_layer_num_max + cls_pooling_layer_num_max > valid_pooling_num_max:
            self.cls_pooling_layer_num_max = max(0, valid_pooling_num_max - joint_pooling_layer_num_max)

        # self.joint_conv_layers = nn.ModuleList(
        #     conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size,
        #              joint_stride_size, joint_padding_size, self.joint_n_output_channels,
        #              self.n_joint_pooling, actiFunc, useBN=self.use_batch_norm))
        
        joint_init_channel_conv_net = conv2d_net(
            input_shape = (self.input_channel_num, self.input_sector_num, self.input_frame_num), 
            n_conv_layers = 1, 
            n_output_channels = self.joint_init_conv_channel_num,
            conv_size = self.joint_conv_kernel_size,
            stride_size = joint_conv_stride_size, 
            padding_size = joint_conv_padding_size,                  
            pooling_method = self.pooling_method, 
            pooling_kernel_size = 2,
            pooling_layers_max_num = None if self.joint_pooling_layer_num_max is None else self.joint_pooling_layer_num_max - 1,
            activation_func = activation_func, 
            batchnorm = self.use_batch_norm, 
            batchnorm_at_end = True, 
            activation_at_end = True)
        
        # print("joint_init_channel_conv_net['output_shape']", joint_init_channel_conv_net['output_shape'])
        joint_contracting_path_conv_net = conv2d_net(
            input_shape = joint_init_channel_conv_net['output_shape'], 
            n_conv_layers = self.joint_conv_layer_num - 1, 
            n_output_channels = self.joint_conv_output_channel_num,
            conv_size = self.joint_conv_kernel_size,
            stride_size = joint_conv_stride_size, 
            padding_size = joint_conv_padding_size,                  
            pooling_method = self.pooling_method, 
            pooling_kernel_size = 2,
            pooling_layers_max_num = None if self.joint_pooling_layer_num_max is None else self.joint_pooling_layer_num_max - 1,
            activation_func = activation_func, 
            batchnorm = self.use_batch_norm, 
            batchnorm_at_end = True, 
            activation_at_end = True)
        
        self.joint_layers_ModuleList = nn.ModuleList(joint_init_channel_conv_net['layers'] + joint_contracting_path_conv_net['layers'])
        # joint_last_conv_layer = get_last_conv_channels_num(joint_conv_layers)
        # joint_pooling_layer_num = get_pooling_layer_num(joint_last_conv_layer)    
        
        # Regression Conv layers
        self.reg_conv_kernel_size = self.paras.get('reg_conv_kernel_size', 3)        
        # self.reg_conv_channel_num = self.paras.get('reg_conv_channel_num', None)
        self.reg_conv_output_channel_num = self.paras.get('reg_conv_output_channel_num', None)        
        self.reg_linear_layer_num = self.paras.get('reg_linear_layer_num', 3)
        self.reg_linear_layer_inner_dim = self.paras.get('reg_linear_inner_layer_dim', None)
        self.reg_output_dim = self.paras.get('reg_output_dim', 18)
        self.reg_output_additional_dim = self.paras.get('reg_output_additional_dim', False)
        self.reg_normalize_layer = self.paras.get('reg_normalize_layer', 'leaky_relu')

        reg_padding_size = self.reg_conv_kernel_size // 2
        reg_conv_stride = 1
        
        # print('reg_net...')
        reg_net = conv_fc_net(
            conv_input_shape = joint_contracting_path_conv_net['output_shape'], 
            linear_output_dim = self.reg_output_dim, 
            conv_layer_num = self.reg_conv_layer_num,
            conv_output_channel_num = self.reg_conv_output_channel_num,
            conv_kernel_size = self.reg_conv_kernel_size,
            conv_stride_size = reg_conv_stride, 
            conv_padding_size = reg_padding_size, 
            conv_pooling_method = self.pooling_method, 
            conv_pooling_kernel_size = 2,
            conv_pooling_layers_max_num = self.reg_pooling_layer_num_max,
            activation_func = activation_func, 
            batchnorm = self.use_batch_norm, 
            linear_layer_num = self.reg_linear_layer_num,
            linear_inner_dim = self.reg_linear_layer_inner_dim,
            simple_last_linear_layer = True)
        
        # print('reg_net!')
        reg_layers = reg_net['layers']
        
        if self.reg_normalize_layer is None:
            pass
        elif self.reg_normalize_layer.lower() in ['leaky_relu', 'leaky_relu_17']:
            reg_layers.append(shift_Leaky_ReLU(shift=17))
        else:
            raise ValueError('Unsupported reg normalize layer: ', self.reg_normalize_layer)
        
        if self.reg_output_additional_dim:
            reg_layers.append(View(-1, 1, self.linear_output_dim))
        
        self.reg_layers_ModuleList = nn.ModuleList(reg_layers)
        
        

        # Classification Conv Layers        
        self.cls_conv_kernel_size = self.paras.get('cls_conv_kernel_size', self.joint_conv_kernel_size)
        
        # self.cls_n_conv_channels = self.paras.get('cls_n_conv_channels', self.joint_n_conv_channels)
        self.cls_conv_output_channel_num = self.paras.get('cls_conv_output_channel_num', None)        
        self.cls_linear_layer_num = self.paras.get('cls_linear_layer_num', 3)
        self.cls_linear_layer_inner_dim = self.paras.get('cls_linear_inner_layer_dim', None)
        self.cls_output_dim = self.paras.get('cls_output_dim', 18)
        self.cls_force_onehot = self.paras.get('cls_force_onehot', True)
        self.cls_class_normlize_layer = self.paras.get('cls_class_normlize_layer', 'softmax')

        cls_padding_size = self.cls_conv_kernel_size // 2
        cls_conv_stride = 1
        
        cls_net = conv_fc_net(
            conv_input_shape = joint_contracting_path_conv_net['output_shape'], 
            linear_output_dim = self.cls_output_dim, 
            conv_layer_num = self.cls_conv_layer_num,
            conv_output_channel_num = self.cls_conv_output_channel_num,
            conv_kernel_size = self.cls_conv_kernel_size,
            conv_stride_size = cls_conv_stride, 
            conv_padding_size = cls_padding_size, 
            conv_pooling_method = self.pooling_method, 
            conv_pooling_kernel_size = 2,
            conv_pooling_layers_max_num = self.cls_pooling_layer_num_max,
            activation_func = activation_func, 
            batchnorm = self.use_batch_norm, 
            linear_layer_num = self.cls_linear_layer_num,
            linear_inner_dim = self.cls_linear_layer_inner_dim,
            simple_last_linear_layer = True)
    
        cls_layers = cls_net['layers']
        if self.cls_force_onehot or self.classes_num > 2:
            # Reshape data to (batch_size, class_num, sector_num)
            cls_layers.append(View((-1, self.classes_num, self.cls_output_dim // self.classes_num)))            
            if self.cls_class_normlize_layer.lower() == 'softmax':
                cls_layers.append(nn.Softmax(dim = -2))
            elif self.cls_class_normlize_layer.lower() == 'log softmax':
                cls_layers.append(nn.LogSoftmax(dim=-2))
        else:
            cls_layers.append(nn.LeakyReLU())
    
        self.cls_layers_ModuleList = nn.ModuleList(cls_layers)

    def set_input_types(self, input_types, input_tags):
        # if type(input_types) is str:
        #     self.strainmat_type = input_types
        # elif type(input_types) is list:
        #     self.strainmat_type = get_data_type_by_category('strainmat', input_types)
        for data_type, data_tag in zip(input_types, input_tags):
            if data_tag.lower() in ['strainmat']:
                self.strainmat_type = data_type
            else:
                raise ValueError('Unsupported tag: ', data_tag)

    # def set_output_types(self, output_types, reg_category='TOS', cls_category='sector_label'):
    def set_output_types(self, output_types, output_tags):
        for data_type, data_tag in zip(output_types, output_tags):
            if data_tag.lower() in ['reg', 'regression']:
                self.reg_type = data_type
            elif data_tag.lower() in ['cls', 'classification']:
                self.cls_type = data_type
            else:
                raise ValueError('Unsupported tag: ', data_tag)
        
        # self.reg_type = get_data_type_by_category(reg_category, output_types)
        # # self.reg_type = get_data_type('fit_coefs', output_types)
        # self.cls_type = get_data_type_by_category(cls_category, output_types)

    @autocast()
    def forward(self, input_dict):
        # print('AAA')
        # enable_autocast = False
        # enable_autocast = True
        if type(input_dict) is dict:
            x = input_dict[self.strainmat_type]
        else:
            x = input_dict
            
        # x = x.float32()
        # with autocast(enabled=enable_autocast):        
        for layer in self.joint_layers_ModuleList:
            # print(x.dtype)
            x = layer(x)

        x_cls = x
        for layer in self.cls_layers_ModuleList:
            x_cls = layer(x_cls)

        # x_cls = x_cls.reshape(x.shape[0], self.n_classes, -1)
        # # xls: [N, n_clasees, n_sectors]
        # x_cls = nn.Softmax(dim=1)(x_cls)
        # print(x_cls.shape)

        x_reg = x
        for layer in self.reg_layers_ModuleList:
            x_reg = layer(x_reg)
        # x_reg = nn.LeakyReLU(inplace=True)(x_reg - 17) + 17
        # x_reg[::2] = torch.round(x_reg[::2])
        if type(input_dict) is dict:
            return {self.reg_type: x_reg, self.cls_type: x_cls}
        else:
            return x_reg, x_cls
        
    
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
                
        
        following_net = conv_fc_net(
            conv_input_shape = init_net['output_shape'],
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
        x = input_data[self.input_type]  # .cuda()        
        for layer in self.layers:
            x = layer(x)        
        
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
        self.reg_output_dim = self.paras.get('reg_output_dim', 256)
        self.classes_num = self.paras.get('classes_num', 1)
        self.activation_func = self.paras.get('activation_func', 'ReLU')
        self.force_onehot = self.paras.get('force_onehot', False)
        self.last_relu_layer = self.paras.get('last_relu_layer', True)
        

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
                
        
        following_net = conv_fc_net(
            conv_input_shape = init_net['output_shape'],
            linear_output_dim = self.reg_output_dim, 
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
        
        if self.last_relu_layer:
            layers += [shift_Leaky_ReLU(17)]
        else:
            layers += []                
            
        if self.force_onehot or self.classes_num > 2:
            # Reshape data to (batch_size, class_num, sector_num)
            layers.append(View((-1, self.classes_num, self.reg_output_dim // self.classes_num)))
            
        self.layers = nn.ModuleList(layers)

    @autocast()
    def forward(self, input_data: dict or torch.Tensor):        
        if type(input_data) is torch.Tensor:
            x = input_data
        else:        
            x = input_data[self.input_type]  # .cuda()
        
        for layer in self.layers:
            x = layer(x)
        
        # x = x.reshape(x.shape[0], self.n_classes, -1)
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
    
    # def set_input_types(self, input_types):
    #     if type(input_types) is str:
    #         self.input_type = input_types
    #     elif type(input_types) is list:
    #         self.input_type = get_data_type_by_category('strainmat', input_types)

    # def set_output_types(self, output_types):
    #     if type(output_types) is str:
    #         self.output_types = output_types
    #     elif type(output_types) is list:
    #         for target_data_category in ['sector_dist_map', 'sector_value']:
    #             output_type = get_data_type_by_category(target_data_category, output_types)                     
    #             if output_type is not None:
    #                 self.output_types = output_type
    
    def set_input_types(self, input_types, input_tags):
        # if type(input_types) is str:
        #     self.strainmat_type = input_types
        # elif type(input_types) is list:
        #     self.strainmat_type = get_data_type_by_category('strainmat', input_types)
        for data_type, data_tag in zip(input_types, input_tags):
            if data_tag.lower() in ['strainmat']:
                self.input_type = data_type
            else:
                raise ValueError('Unsupported tag: ', data_tag)

    # def set_output_types(self, output_types, reg_category='TOS', cls_category='sector_label'):
    def set_output_types(self, output_types, output_tags):
        for data_type, data_tag in zip(output_types, output_tags):
            if data_tag.lower() in ['reg', 'regression']:
                self.output_types = data_type
            else:
                raise ValueError('Unsupported tag: ', data_tag)
                
if __name__ == '__main__':
    # Common settings
    test_network = 'NetStrainMat2Reg'
    debug_config = {
        'paras':{
            'input_channel_num': 1,
            'init_conv_channel_num': 16,
            'input_sector_num': 128,
            'input_frame_num': 64,            
            'pooling_method': 'maxpooling',            
            'classes_num': 2,
            'force_onehot': True
            }
        }
    
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
        
        debug_config['paras']['conv_layer_num'] = 3
        debug_config['paras']['conv_output_channel_num'] = None
        debug_config['paras']['linear_layer_num'] = 3
        debug_config['paras']['cls_output_dim'] = 256        
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
        
        
    elif test_network == 'NetStrainMat2Reg':
        from torch.cuda.amp import autocast, GradScaler
        
        debug_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        debug_batch_size = 64
        # debug_input_types = ['strainMatFullResolutionSVD']
        # debug_input_tags = ['strainmat']
        # debug_input_shapes = [(debug_batch_size, 1, 128, 64)]
        # debug_input_data = {}
        # for debug_input_idx in range(len(debug_input_types)):
        #     debug_input_data[debug_input_types[debug_input_idx]] = torch.rand(debug_input_shapes[debug_input_idx]).to(debug_device)
        debug_input_data_info = [
            {
                'type': 'strainMatFullResolutionSVD',
                'config': {},
                'tag': 'strainmat'
                }
            ]
        debug_input_data_types = [term['type'] for term in debug_input_data_info]
        debug_input_data = {'strainMatFullResolutionSVD': torch.rand((debug_batch_size, 1, 128, 64)).to(debug_device)}
            
        # debug_target_types = ['TOS126']
        # debug_target_shapes = [(debug_batch_size, 128)]
        # debug_target_data = {}
        debug_output_data_info = [
            {
                'type': 'TOS126',
                'config': {},
                'tag': 'reg'
                }
            ]
        debug_output_data_types = [term['type'] for term in debug_output_data_info]
        debug_target_data = {'TOS126': torch.rand((debug_batch_size, 128)).to(debug_device)}
        
        
        # for debug_target_idx in range(len(debug_target_types)):
        #     debug_target_data[debug_target_types[debug_target_idx]] = torch.rand(debug_target_shapes[debug_target_idx]).to(debug_device)
        
        debug_config['paras']['conv_layer_num'] = 3
        debug_config['paras']['conv_output_channel_num'] = None
        debug_config['paras']['linear_layer_num'] = 3
        debug_config['paras']['reg_output_dim'] = 128
        debug_config['paras']['classes_num'] = 1
        debug_config['paras']['force_onehot'] = False
        debug_network = NetStrainMat2Reg(debug_config).to(debug_device)
        
        debug_network.set_input_types(debug_input_data_types, [term['tag'] for term in debug_input_data_info])
        debug_network.set_output_types(debug_output_data_types, [term['tag'] for term in debug_output_data_info])
        
        debug_optimizer = torch.optim.Adam(debug_network.parameters(), lr=1e-4,
                                              weight_decay=1e-5)
        
        debug_scaler = GradScaler(enabled=True)
        
        debug_loss = 0
        with autocast():
            debug_output_data = debug_network(debug_input_data)
            for debug_data_type_idx in range(len(debug_target_types)):
                debug_loss += torch.nn.MSELoss()(debug_output_data[debug_target_types[debug_data_type_idx]], debug_target_data[debug_target_types[debug_data_type_idx]])
                # debug_loss += nn.BCEWithLogitsLoss()(debug_output_data[debug_target_types[debug_data_type_idx]], debug_target_data[debug_target_types[debug_data_type_idx]])
        
        # loss.backward()                
        # optimizer.step()
        
        debug_scaler.scale(debug_loss).backward()
        debug_scaler.step(debug_optimizer)
        debug_scaler.update()
        