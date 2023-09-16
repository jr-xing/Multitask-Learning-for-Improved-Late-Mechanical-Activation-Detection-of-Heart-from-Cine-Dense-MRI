# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:32:14 2021

@author: Jerry Xing
"""
import torch
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
from utils.data import get_data_type_by_category
def conv_horipooling_bn_acti_layer(conv_in_channels, conv_out_channels, conv_size, conv_stride, conv_padding,
                               actiFunc, useBN = False):
    layers = [
        nn.Conv2d(conv_in_channels, conv_out_channels, conv_size, 
                       stride=conv_stride, padding=conv_padding),                
        ]
    if useBN:
        layers += [nn.BatchNorm2d(conv_out_channels)]
        
    layers += [
        actiFunc,
        nn.MaxPool2d((1,2), stride = (1,2))
        ]
    return layers

def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc, useBN = False):
    linear_layers = [nn.Flatten(), 
                   nn.Linear(linear_input_feature_dim, linear_inner_feature_dim),
                   actiFunc]
    if useBN:
        linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
        
    for inner_layer_idx in range(n_linear_layers - 2):
        linear_layers += [nn.Linear(linear_inner_feature_dim, linear_inner_feature_dim), actiFunc]
        if useBN:
            linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
            
    linear_layers += [nn.Linear(linear_inner_feature_dim, linear_output_feature_dim)]
    
    return linear_layers

class NetStrainMat2PolycoeffCls(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2PolycoeffCls, self).__init__()
        # Joint conv layers
        self.paras = config.get('paras', None)
        self.n_input_channels       = self.paras.get('n_input_channels', 1)
        self.n_sectors_in           = self.paras.get('n_sectors_in', 18)
        self.n_sectors_out          = self.paras.get('n_sectors_out', 18)
        self.n_frames               = self.paras.get('n_frames',  25)
        self.use_batch_norm         = self.paras.get('batch_norm',  False)
        self.joint_n_conv_layers    = self.paras.get('joint_n_conv_layers',  6)
        self.joint_n_conv_channels  = self.paras.get('joint_n_conv_channels',  16)
        self.joint_conv_size        = self.paras.get('joint_conv_size',  3)
        self.joint_n_output_channels= self.paras.get('joint_n_output_channels',  self.joint_n_conv_channels)
        self.n_joint_pooling = 2
        self.n_classes              = self.paras.get('n_classes', 2)
        
        joint_padding_size = self.joint_conv_size // 2
        joint_stride_size  = 1
        
        useBN = self.use_batch_norm
        # actiFunc = nn.Sigmoid()
        # actiFunc = nn.ReLU(True)
        actiFunc = nn.LeakyReLU(True)
                
        # self.joint_conv_layers = nn.ModuleList(conv_net(self.n_input_channels, self.joint_n_conv_layers, self.joint_n_conv_channels, self.joint_conv_size, 
        #            joint_stride_size, joint_padding_size, self.joint_n_output_channels,
        #            self.n_joint_pooling, actiFunc,useBN = self.use_batch_norm))
        if self.joint_n_conv_layers > 1:
            joint_conv_layers = conv_horipooling_bn_acti_layer(1, self.joint_n_conv_channels, self.joint_conv_size,
                                               joint_stride_size, joint_padding_size, actiFunc, self.use_batch_norm)
        
            for _ in range(self.joint_n_conv_layers - 2):
                joint_conv_layers += conv_horipooling_bn_acti_layer(self.joint_n_conv_channels, self.joint_n_conv_channels, self.joint_conv_size,
                                               joint_stride_size, joint_padding_size, actiFunc, self.use_batch_norm)
            
            joint_conv_layers +=  conv_horipooling_bn_acti_layer(self.joint_n_conv_channels, self.joint_n_output_channels, self.joint_conv_size,
                                           joint_stride_size, joint_padding_size, actiFunc, self.use_batch_norm)
        else:
            joint_conv_layers = conv_horipooling_bn_acti_layer(1, self.joint_n_output_channels, self.joint_conv_size,
                                               joint_stride_size, joint_padding_size, actiFunc, self.use_batch_norm)
        self.joint_conv_layers = nn.ModuleList(joint_conv_layers)
        
        # Regression Linear layers
        self.reg_output_dim        = (self.paras.get('degree', 10) + 1) * 2
        self.reg_n_linear_layers   = self.paras.get('reg_n_linear_layers', 3)
                
        reg_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.joint_n_output_channels // (1*(2**self.joint_n_conv_layers))
        reg_linear_inner_feature_dim = reg_linear_input_feature_dim# // 2
        reg_linear_output_feature_dim = self.reg_output_dim
        
        reg_linears = fcn(self.reg_n_linear_layers, reg_linear_input_feature_dim, reg_linear_inner_feature_dim, reg_linear_output_feature_dim, 
                          actiFunc, useBN = self.use_batch_norm)                
        self.reg_layers = nn.ModuleList(reg_linears)
        
        # Classification Linear Layers
        self.cls_n_linear_layers   = self.paras.get('cls_n_linear_layers', 3)                
        cls_linear_input_feature_dim = self.n_sectors_in*self.n_frames*self.joint_n_output_channels // (1*(2**self.joint_n_conv_layers))
        cls_linear_inner_feature_dim = cls_linear_input_feature_dim# // 2
        
        cls_linears = fcn(self.cls_n_linear_layers, cls_linear_input_feature_dim, cls_linear_inner_feature_dim, self.n_classes*self.n_sectors_out, 
                          actiFunc, useBN = self.use_batch_norm)
        
        self.cls_layers = nn.ModuleList(cls_linears)                
    
    def set_input_types(self, input_types):
        if type(input_types) is str:
            self.strainmat_type = input_types
        elif type(input_types) is list:
            self.strainmat_type = get_data_type('strainmat', input_types)
    
    def set_output_types(self, output_types):
        # self.coeff_type = get_data_type('fit_coefs', output_types)
        self.TOS_type = get_data_type('TOS', output_types)
        self.classify_label_type = get_data_type('sector_label', output_types)
        
    def set_device(self, device):
        self.device = device
    
    @autocast()
    def forward(self, input_dict):
        x = input_dict[self.strainmat_type]
        for layer in self.joint_conv_layers:
            x = layer(x)
        # print('shape after joint: ', x.shape)
        
        x_cls = x
        for layer in self.cls_layers:
            x_cls = layer(x_cls)        
            
        x_cls = x_cls.reshape(-1, self.n_classes, self.n_sectors_out)
        # xls: [N, n_clasees, n_sectors]
        x_cls = nn.Softmax(dim = 1)(x_cls)
        
        x_reg = x
        for layer in self.reg_layers:
            x_reg = layer(x_reg)
        
        # print(x_reg.shape)
        coefs_mags = x_reg[:,::2]
        coefs_vals = x_reg[:,1::2]
        coefs = coefs_vals * 10**coefs_mags
        
        # reg = np.polyval(coefs, torch.arange(len(coefs_mags)))
        # print(coefs.shape)
        reg = torch.zeros(coefs.shape[0], self.n_sectors_out).to(self.device)
        for degree in range(self.reg_output_dim//2):
            # print((torch.arange(self.n_sectors_out) ** degree).repeat(coefs.shape[0],1).shape)
            # print(coefs[:, degree].view(coefs.shape[0],1).shape)
            # print((torch.arange(self.n_sectors_out) ** degree).repeat(coefs.shape[0],1).shape)
            # print(torch.arange(self.n_sectors_out).to(self.device).device)
            reg += coefs[:, degree].view(coefs.shape[0],1) * (torch.arange(self.n_sectors_out).to(self.device) ** degree).repeat(coefs.shape[0],1) 
        
        # return {self.coeff_type: reg, self.classify_label_type: x_cls}
        return {self.TOS_type: reg, self.classify_label_type: x_cls}
    
if __name__ == '__main__':
    print('TEST POLYFIT NETWORK')
    config = {
        'paras':{
            'n_sectors_in': 128,
            'n_sectors_out': 128,
            'n_frames': 64
            }}
    network = NetStrainMat2PolycoeffCls(config)
    network.set_input_types('strainmat')
    network.set_output_types(['TOS', 'late_acti_label'])
    # network.set_output_types(['polyfit_coefs', 'late_acti_label'])
    testdatum = torch.zeros(2,1,128,64)
    testoutput = network({'strainmat':testdatum})
    