# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:22:26 2021

@author: remus
"""
import torch
from torch import nn
from utils.data import get_data_type_by_category, get_data_info_by_tag
from icecream import ic
def evaluate(data: dict, data_types_to_eval: list, output_info: list, method: str = 'MSE', paras: list or dict = {}, print_loss_details = False):
    # Evaluate with single metric
    # data: dictionary of single data point
    # data_types_to_eval: list of strings
    # method
    # paras
    if method == 'MSE':
        if len(data_types_to_eval) == 1:
            # print(data_types_to_eval)
            # print(data['net_output'][data_types_to_eval[0]].shape)
            # print(data[data_types_to_eval[0]].shape)
            # return nn.MSELoss()(data['output'][data_types_to_eval[0]], data[data_types_to_eval[0]])
            loss_reg = nn.MSELoss()(data['net_output'][data_types_to_eval[0]], data[data_types_to_eval[0]])
            return loss_reg, loss_reg
        else:
            raise ValueError('Cannnot eva more than 1 types')
    elif method == 'multitask-reg-cls':
        # Classification loss
        cls_data_type = get_data_info_by_tag('cls', output_info)['type']
        n_classes = data[cls_data_type].shape[-2]        
        cls_eval_paras = [term for term in paras if term['target_tag'] == 'cls'][0]
        cls_weight = cls_eval_paras.get('weight', 1e1)
        if cls_eval_paras['type'].lower() in ['cross entropy', 'ce']:
            if n_classes == 2:
                cls_loss = nn.BCEWithLogitsLoss()(data[cls_data_type], data['net_output'][cls_data_type]) 
            else:
                cls_loss = nn.CrossEntropyLoss()(data['net_output'][cls_data_type], torch.argmax(data[cls_data_type], axis=-2)) 
        elif cls_eval_paras['type'].lower() in ['negative log likelihood', 'nll']:
            cls_loss = nn.NLLLoss()(data['net_output'][cls_data_type], torch.argmax(data[cls_data_type], axis=-2)) 
        else:
            raise ValueError('Unsupported classification loss: ', cls_eval_paras['type'])
        
        # Regression loss
        reg_data_type = get_data_info_by_tag('reg', output_info)['type']
        reg_eval_paras = [term for term in paras if term['target_tag'] == 'reg'][0]
        if reg_eval_paras['type'].lower() in ['mse']:
            reg_loss = nn.MSELoss(reduction='mean')(torch.squeeze(data[reg_data_type]), torch.squeeze(data['net_output'][reg_data_type]))
        else:
            raise ValueError('Unsupported regressionn loss: ', reg_eval_paras['type'])
        
        cls_weight = cls_eval_paras.get('weight', 1e1)
        reg_weight = reg_eval_paras.get('weight', 1e0)
        # print(torch.squeeze(data[reg_data_type]).shape, torch.squeeze(data['net_output'][reg_data_type]).shape)
        total_loss = cls_weight * cls_loss + reg_weight * reg_loss
        return total_loss, reg_loss
        
    elif method == 'multitask-reg-clsDistMap':
        TOS_type = get_data_type_by_category('TOS', data_types_to_eval)
        dist_map_type = get_data_type_by_category('sector_dist_map', data_types_to_eval)
        # print(data_types_to_eval)
        # print('dist_map_type:', dist_map_type)
        loss_reg = nn.MSELoss(reduction='mean')(data[TOS_type], data['net_output'][TOS_type])
        # loss_cls = paras.get('cls_weight', 1e3)*nn.MSELoss(reduction='mean')(data[dist_map_type], data['net_output'][dist_map_type])
        # print(data[dist_map_type].shape, data['net_output'][dist_map_type].shape)
        # print(data['net_output'].keys())
        loss_cls = nn.MSELoss(reduction='mean')(data[dist_map_type], data['net_output'][dist_map_type])
        # print('Loss reg / cls: ', loss_reg, loss_cls)
        return loss_reg + loss_cls, loss_reg
    elif method == 'cls':
        loss_cls_type = paras.get('type', 'Cross Entropy')
        classifiy_label_type = get_data_type_by_category('sector_label', data_types_to_eval)
        if classifiy_label_type is None:
            classifiy_label_type = get_data_type_by_category('data_label', data_types_to_eval)
        # print(data[classifiy_label_type].dtype, data['net_output'][classifiy_label_type].dtype)
        n_classes = data[classifiy_label_type].shape[-2]
        if loss_cls_type == 'Cross Entropy':
            # print(data[classifiy_label_type],  data['net_output'][classifiy_label_type])
            # return
            if n_classes == 2:
                loss_cls = nn.BCEWithLogitsLoss()(data[classifiy_label_type], data['net_output'][classifiy_label_type]) 
            elif n_classes == 3:
                loss_cls = nn.CrossEntropyLoss()(data['net_output'][classifiy_label_type], torch.argmax(data[classifiy_label_type], axis=-2)) 
            # print(loss_cls)
        elif loss_cls_type == 'MSE':
            pass            
        elif loss_cls_type.lower() == 'negative log likelihood':
            loss_cls = nn.NLLLoss()(data['net_output'][classifiy_label_type], torch.argmax(data[classifiy_label_type], axis=-2)) 
        return loss_cls, loss_cls
    elif method == 'reg':
        reg_eval_paras = [term for term in paras if term['target_tag'] == 'reg'][0]
        
        loss_reg_type = reg_eval_paras.get('type', 'MSE')
        reg_data_type = get_data_info_by_tag('reg', output_info)['type']
        # ic(loss_reg_type)
        if loss_reg_type == 'MSE':
            # ic(data['net_output'][data_types_to_eval[0]].dtype)
            # ic(data[data_types_to_eval[0]].dtype)
            # loss_reg = nn.MSELoss()(data['net_output'][data_types_to_eval[0]], data[data_types_to_eval[0]])
            reg_loss = nn.MSELoss(reduction='mean')(torch.squeeze(data[reg_data_type]), torch.squeeze(data['net_output'][reg_data_type]))
            return reg_loss, reg_loss
    else:
        raise ValueError(f'Evaluation Method {method} not supported')
    