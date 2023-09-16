# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:42:25 2021

@author: remus
"""
import json

def get_default_config(exp_type='strainmat_to_TOS'):
    if exp_type == 'strainmat_to_TOS':
        aug_config = [{
            'method': 'shift_sector',
            'shift_amount': list(range(-9, 9))
        }]
        data_config = {
            'filename': 'D://dataFull-201-2020-12-23-Jerry.npy',
            'input_types': ['strainMat'],
            'output_types': ['TOS18_Jerry'],
            'train_test_split': {'method': 'set_test_by_patient',
                                 'paras': {'test_patient_names': ['SET01-CT11', 'SET02-CT28', 'SET03-EC21']}
                                 # 'paras':{'test_patient_names': ['SET01-CT11', 'SET02-CT28', 'SET03-EC21', 'SET03-UP36']}
                                 },
            'augmentation': aug_config
        }
        net_config = {
            'type': 'NetStrainMat2TOS',
            'paras': {
                'n_sector': -1,
                'n_frames': -1,
                'n_conv_layers': 4,
                'n_conv_channels': 8,
                'conv_size': 3,
                'conv_stride': 1
            }
        }
        training_config = {
            'epochs_num': 1000,
            'batch_size': 100,
            'learning_rate': 1e-4,
            'report_per_epochs': 50,
            'training_check': False,
            'valid_check': True,
        }
        eval_config = {
            'method': 'MSE',
            'paras': {}
        }
        regularize_config = {
            'method': 'L1',
            'paras': {'weight': 1}}
        config = {
            'data': data_config,
            'net': net_config,
            'training': training_config,
            'eval': eval_config,
            'regularization': regularize_config
        }

    # elif exp_type in ['multitask_reg_cls', 'multitask-reg-cls']:
    elif exp_type in ['multitask_reg_cls', 'multitask-reg-cls']:
        aug_config = [{
            'method': 'shift_sector',
            'shift_amount': list(range(-9, 9))
        }]
        data_config = {
            'filename': 'D://dataFull-201-2020-12-23-Jerry.npy',
            'input_info': [{'type': 'strainMat', 'config': {}}],
            'output_info': [{'type': 'TOS18', 'config': {}},
                            {'type': 'late_acti_label', 'config': {}}
                            ],
            # 'TOS_type': 'TOS18_Jerry',
            'train_test_split': {'method': 'set_test_by_patient',
                                 'paras': {'test_patient_names': ['SET01-CT11', 'SET02-CT28', 'SET03-EC21']}
                                 #  'paras':{'test_patient_names': ['SET01-CT11', 'SET02-CT28', 'SET03-EC21', 'SET03-UP36']}
                                 },
            'augmentation': aug_config,
            'scar_free': True
        }
        net_config = {
            'type': 'NetStrainMat2ClsReg',
            'paras': {
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
        training_config = {
            'epochs_num': 1000,
            'batch_size': 100,
            'learning_rate': 1e-4,
            'report_per_epochs': 50,
            'training_check': False,
            'valid_check': True,
        }
        eval_config = {
            'method': 'multitask-reg-cls',
            'paras': {
                # 'cls_weight': 1e-2,
                'cls_weight': 1e1,
            }
        }
        regularize_config = {
            'method': 'L1',
            'paras': {'weight': 1e-1}}
        config = {
            'data': data_config,
            'net': net_config,
            'training': training_config,
            'eval': eval_config,
            'regularization': regularize_config
        }
    elif exp_type == 'multitask_reg_clsDistMap':
        config = get_default_config('multitask_reg_cls')
        config['data']['output_info'][1] = {'type': 'late_acti_dist_map', 'config': {}}
        config['data']['augmentation'][0]['shift_amount'] = list(range(-31, 31, 5))
        config['net']['type'] = 'NetStrainMat2ClsDistMapReg'
        config['eval']['method'] = 'multitask-reg-clsDistMap'
        config['eval']['paras']['class_type'] = 'late_acti_dist_map'

    # elif exp_type == 'multitask_reg_strain_type_cls':
    #     config = get_default_config('multitask_reg_cls')
    #     config['data']['output_types'][1] = 'strain_curve_type_label'
    #     config['data']['augmentation'][0]['shift_amount'] = list(range(-31, 31 ,5))
    #     config['net']['type'] = 'NetStrainMat2Cls'
    #     config['eval']['method'] = 'multitask-reg-cls'
    #     # config['eval']['paras']['generate_method'] = 'strain-based'

    # elif exp_type == 'multitask_reg_strain_type_dist_map':
    #     config = get_default_config('multitask_reg_strain_type_cls')
    #     config['data']['output_types'][1] = 'strain_curve_type_dist_map'
    #     config['data']['augmentation'][0]['shift_amount'] = list(range(-31, 31 ,5))
    #     config['net']['type'] = 'NetStrainMat2ClsDistMapReg'
    #     config['eval']['method'] = 'multitask-reg-clsDistMap'

    elif exp_type == 'late_acti_cls':
        aug_config = [{
            'method': 'shift_sector',
            'shift_amount': list(range(-31, 31, 5))
        }]
        data_config = {
            'filename': 'D://dataFull-201-2020-12-23-Jerry.npy',
            'input_types': ['strainMat'],
            'output_types': ['TOS18_Jerry', 'late_acti_label'],
            'train_test_split': {'method': 'set_test_by_patient',
                                 'paras': {'test_patient_names': ['SET01-CT11', 'SET02-CT28', 'SET03-EC21']}
                                 },
            'augmentation': aug_config
        }
        net_config = {
            'type': 'NetStrainMat2ClsReg',
            'paras': {
                'n_sector': 18,
                'n_frames': 25
            }
        }
        training_config = {
            'epochs_num': 1000,
            'batch_size': 100,
            'learning_rate': 1e-4,
            'report_per_epochs': 50,
            'training_check': False,
            'valid_check': True,
        }
        eval_config = {
            'method': 'multitask-reg-cls',
            'paras': {
                'class_type': 'late_acti_label',  # or 'lateActi-scar-normal'
                'generate_method': 'TOS_based',
                'threshold': 17
            }
        }
        regularize_config = {
            'method': 'L1',
            'paras': {'weight': 1}}
        config = {
            'data': data_config,
            'net': net_config,
            'training': training_config,
            'eval': eval_config,
            'regularization': regularize_config
        }
    elif exp_type == 'cls':
        config = get_default_config('multitask_reg_cls')
        config['data']['input_info'] = [{'type': 'strainMatFullResolution', 'config': {}}]
        config['data']['output_info'] = [{'type': 'scar_sector_label', 'config': {}}]
        # config['data']['augmentation'][0]['shift_amount'] = list(range(-31, 31, 5))
        config['data']['augmentation'] = []
        config['net']['type'] = 'NetStrainMat2Cls'
        config['net']['paras'] = {
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
            'force_onehot': True}
        config['eval']['method'] = 'cls'
        config['eval']['paras']['type'] = 'Cross Entropy'
    elif exp_type == 'reg':
        config = get_default_config('multitask_reg_cls')
        config['data']['input_info'] = [{'type': 'strainMatFullResolution', 'config': {}}]
        config['data']['output_info'] = [{'type': 'scar_sector_distmap', 'config': {}}]
        # config['data']['augmentation'][0]['shift_amount'] = list(range(-31, 31, 5))
        config['data']['augmentation'] = []
        config['net']['type'] = 'NetStrainMat2Reg'
        config['eval']['method'] = 'reg'
        config['eval']['paras']['type'] = 'MSE'
    else:
        raise ValueError(f'Unsupported experiment type: {exp_type}')

    return config


from utils.augmentation import parse_augment_argument_json


# def update_iotype_argument(data_arguments: str):
#     io_config = []
#     for data_argument in data_arguments.split('+'):
#         data_argument_split = data_argument.split('=')
#         if len(data_argument_split) > 1:
#             data_type, data_config_str = data_argument_split
#         else:
#             data_type = data_argument_split[0]
#             data_config_str = ''

#         if data_type in ['scar-AHA-step']:
#             scar_region_threshold = int(data_config_str)
#             io_config.append({
#                 'type': data_type,
#                 'config': {
#                     'scar_region_threshold': scar_region_threshold
#                 }
#             })
#         else:
#             io_config.append({
#                 'type': data_type,
#                 'config': {}
#             })
#     return io_config

def update_iotype_argument(data_arguments: str):
    # <tag1>:<type1>=<parameters>+<tag2>:<typ2>=<parameters>
    io_config = []
    for data_argument in data_arguments.split('+'):
        data_tag = data_argument.split(':')[0]
        data_type_argument_split = data_argument.split(':')[1].split('=')
        if len(data_type_argument_split) > 1:
            data_type, data_config_str = data_type_argument_split
        else:
            data_type = data_type_argument_split[0]
            data_config_str = ''

        if data_type in ['scar-AHA-step']:
            scar_region_threshold = int(data_config_str)
            io_config.append({
                'type': data_type,
                'config': {
                    'scar_region_threshold': scar_region_threshold
                }
            })
        else:
            io_config.append({
                'type': data_type,
                'config': {},
                'tag': data_tag
            })
    return io_config

def modify_config(config=None, exp_type: str = 'strainmat_to_TOS', terms_to_modify: dict = {}):
    if config is None:
        config = get_default_config(exp_type)

    if exp_type in ['strainmat_to_TOS']:
        for key in terms_to_modify.keys():
            if key == 'aug_config':
                config['data']['aug_config'] = terms_to_modify[key]
            elif key == 'data_filename':
                config['data']['filename'] = terms_to_modify[key]
            elif key in ['input_types', 'output_types']:
                config['data'][key] = terms_to_modify[key]
            elif key in ['n_conv_layers', 'n_conv_channels', 'conv_size', 'conv_stride']:
                config['net']['paras'][key] = terms_to_modify[key]
            elif key in ['epochs_num', 'batch_size', 'learning_rate']:
                config['training'][key] = terms_to_modify[key]
            elif key is 'regularize_weight':
                config['regularization']['paras']['weight'] = terms_to_modify[key]

    elif exp_type in ['multitask_reg_cls', 'multitask-reg-cls', 'multitask_reg_clsDistMap', 'cls', 'reg']:
        for key in terms_to_modify.keys():
            print('Updating config: key ', key)

            # Data related
            if key == 'augmentation':
                # update_augment_argument(terms_to_modify[key], config['data']['augmentation'])
                config['data']['augmentation'] = parse_augment_argument_json(terms_to_modify[key])
            elif key =='aug_more_on_data_with_scar':
                config['data']['aug_more_on_data_with_scar'] = terms_to_modify[key]
            # elif key == 'scar_free':
            #     if terms_to_modify[key] == 'True':
            #         config['data']['scar_free'] = True
            #     elif terms_to_modify[key] == 'False':
            #         config['data']['scar_free'] = False
            #     else:
            #         raise ValueError(f'Unsupported: {key}: {terms_to_modify[key]}')
            elif key == 'use_data_with_scar':
                config['data'][key] = terms_to_modify[key]
            elif key == 'remove_sector_label_spikes':
                config['data']['remove_sector_label_spikes'] = terms_to_modify[key]
            elif key in ['input_info', 'output_info']:
                # config['data'][key] = terms_to_modify[key].split('+')
                config['data'][key] = update_iotype_argument(terms_to_modify[key])
            elif key == 'data_filename':
                config['data']['filename'] = terms_to_modify[key]

            # Network Related
            elif key in ['joint_n_conv_layers', 'joint_n_conv_channels', 'joint_conv_size',
                         'reg_n_conv_layers', 'reg_n_conv_channels', 'reg_conv_size', 'reg_n_linear_layers',
                         'cls_n_conv_layers', 'cls_n_conv_channels', 'cls_conv_size', 'cls_n_linear_layers',
                         'activation_func',
                         'batch_norm', 
                         'n_conv_layers', 'n_conv_channels', 'conv_size', 'n_init_conv_channels',
                         'n_linear_layers']:
                config['net']['paras'][key] = terms_to_modify[key]
            elif key in ["input_channel_num","input_sector_num","input_frame_num",
                         "joint_init_conv_channel_num","joint_conv_layer_num","joint_conv_channel_num",
                         "joint_conv_kernel_size","joint_pooling_layer_num_max","joint_conv_output_channel_num",
                         "reg_conv_kernel_size","reg_conv_layer_num","reg_conv_output_channel_num",
                         "reg_pooling_layer_num_max","reg_linear_layer_num","reg_linear_layer_inner_dim",
                         "reg_output_dim","reg_output_additional_dim","reg_normalize_layer",
                         "cls_conv_kernel_size","cls_conv_layer_num","cls_conv_output_channel_num",
                         "cls_pooling_layer_num_max","cls_linear_layer_num",
                         "cls_linear_layer_inner_dim","cls_output_dim",
                         "cls_force_onehot","cls_class_normlize_layer"
                         "pooling_method","activation_func","use_batch_norm","classes_num",
                         "conv_layer_num", "pooling_layer_num_max", "linear_layer_num",
                         "init_conv_channel_num", "last_relu_layer"]:
                config['net']['paras'][key] = terms_to_modify[key]
            # Training related
            elif key in ['epochs_num', 'batch_size', 'learning_rate']:
                config['training'][key] = terms_to_modify[key]

                # Regularization
            elif key == 'regularize_weight':
                config['regularization']['paras']['weight'] = terms_to_modify[key]
            elif key == 'cls_weight':
                config['eval']['paras']['cls_weight'] = float(terms_to_modify[key])
            elif key in ['data_record_filename', 'exp_type']:
                pass
            elif key in ['activation_func']:
                config['paras']['activation_func'] = terms_to_modify[key]     
            elif key in ['eval']:
                config['eval'].update(json.loads(terms_to_modify[key]))
            elif key == 'CAM_target_sector':
                if 'show' not in config.keys():
                    config['show'] = {key: terms_to_modify[key]}
                else:
                    config['show'][key] = terms_to_modify[key]
            else:
                raise ValueError('Unsupported parameter: ', key)
    return config


def get_modified_config(exp_type: str = 'strainmat_to_TOS', terms_to_modify: dict = {}):
    return modify_config(None, exp_type, terms_to_modify)
