# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:08:07 2021

@author: Jerry Xing
"""
import torch
from torch import nn
from modules.networks.strain_networks import NetStrainMat2Cls, NetStrainMat2ClsReg
if __name__ == '__main__':
    # Common settings
    test_network = 'NetStrainMat2ClsReg'
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
        
        
    elif test_network == 'NetStrainMat2ClsReg':
        from torch.cuda.amp import autocast, GradScaler
        
        debug_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        # debug_device = torch.device('cpu')
        
        debug_batch_size = 8
        # debug_input_types = ['strainMatFullResolutionSVD']
        # debug_input_shapes = [(debug_batch_size, 1, 128, 64)]
        # debug_input_data = {}
        # for debug_input_idx in range(len(debug_input_types)):
        #     debug_input_data[debug_input_types[debug_input_idx]] = torch.rand(debug_input_shapes[debug_input_idx]).to(debug_device)
        debug_input_data = {'strainMatFullResolutionSVD': torch.rand((debug_batch_size, 1, 128, 64)).half().to(debug_device)}
        debug_input_data_types = list(debug_input_data.keys())
        debug_input_data_tags = ['strainmat']
            
        # debug_target_types = ['late_activation_sector_label', 'TOS126']
        # debug_target_shapes = [(debug_batch_size, 2, 128), (debug_batch_size, 128)]
        # debug_target_data = {}
        
        
        # for debug_target_idx in range(len(debug_target_types)):
        #     debug_target_data[debug_target_types[debug_target_idx]] = torch.rand(debug_target_shapes[debug_target_idx]).to(debug_device)
        debug_target_data = {
            'late_activation_sector_label': torch.rand((debug_batch_size, 2, 128)).half().to(debug_device),
            'TOS126': torch.rand((debug_batch_size, 128)).half().to(debug_device)
            }
            
        debug_output_data_tags = ['cls', 'reg']
        
        debug_config['paras']['joint_init_conv_channel_num'] = 8
        debug_config['paras']['joint_conv_layer_num'] = 1
        debug_config['paras']['joint_pooling_layer_num_max'] = None
        debug_config['paras']['reg_conv_layer_num'] = 1
        debug_config['paras']['reg_output_dim'] = 128
        debug_config['paras']['reg_pooling_layer_num_max'] = 0
        debug_config['paras']['cls_pooling_layer_num_max'] = None
        debug_config['paras']['cls_output_dim'] = 256
        debug_config['paras']['cls_force_onehot'] = True
        debug_network = NetStrainMat2ClsReg(debug_config).to(debug_device)
        
        debug_network.set_input_types(debug_input_data_types, debug_input_data_tags)
        debug_network.set_output_types(debug_target_data.keys(), debug_output_data_tags)
        
        debug_optimizer = torch.optim.Adam(debug_network.parameters(), lr=1e-4,
                                              weight_decay=1e-5)
        
        debug_scaler = GradScaler(enabled=True)
        
        debug_loss = 0
        with autocast():
            debug_output_data = debug_network(debug_input_data)
            # for debug_data_type_idx in range(len(debug_target_types)):
                # debug_loss += torch.nn.MSELoss()(debug_output_data[debug_target_types[debug_data_type_idx]], debug_target_data[debug_target_types[debug_data_type_idx]])
                # debug_loss += nn.BCEWithLogitsLoss()(debug_output_data[debug_target_types[debug_data_type_idx]], debug_target_data[debug_target_types[debug_data_type_idx]])
            debug_loss += nn.BCEWithLogitsLoss()(debug_target_data['late_activation_sector_label'], debug_target_data['late_activation_sector_label'])
            debug_loss += nn.MSELoss()(debug_target_data['TOS126'], debug_target_data['TOS126'])
        
        # loss.backward()                
        # optimizer.step()
        
        debug_scaler.scale(debug_loss).backward()
        debug_scaler.step(debug_optimizer)
        debug_scaler.update()