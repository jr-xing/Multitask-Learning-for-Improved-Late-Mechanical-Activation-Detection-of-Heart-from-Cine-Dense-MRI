# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 13:40:46 2021

@author: Jerry Xing
"""
import numpy as np
from utils.scar_utils import find_connected_components_binary_1d
def get_target_sector(datum, target_data_type, target_sector_type = 'GT'):
    # target_data_category = get_data_type_category(target_data_type)    
    if target_sector_type == 'GT':        
        target_sector = np.argmax(np.squeeze(datum[target_data_type]))
    elif target_sector_type == 'pred':
        target_sector = np.argmax(np.squeeze(datum[target_data_type+'_pred']))
    elif target_sector_type == 'difference':
        target_sector = np.argmax(np.abs(np.squeeze(datum[target_data_type+'_pred']) - np.squeeze(datum[target_data_type])))
    elif target_sector_type == 'over-pred':
        target_sector = np.argmax(np.squeeze(datum[target_data_type+'_pred']) - np.squeeze(datum[target_data_type]))
    elif target_sector_type == 'under-pred':
        target_sector = np.argmin(np.squeeze(datum[target_data_type+'_pred']) - np.squeeze(datum[target_data_type]))
    elif target_sector_type == 'scar_center':
        scar_regions = find_connected_components_binary_1d((datum['scar_sector_label']).astype(np.int)[0,1,:], order = 'size')[0]
        if len(scar_regions) > 0:
            largest_scar_region_center = scar_regions[0]['center']
        else:
            largest_scar_region_center = 60
        target_sector = largest_scar_region_center
    elif target_sector_type == 'late_activation_center':
        if 'late_activation_sector_label' in datum.keys():
            late_activation_sector_label = datum['late_activation_sector_label']
        else:
            late_activation_sector_label = datum['TOS126'] > 17.01
            
        if late_activation_sector_label.shape[-2] > 1:
            late_activation_sector_label_arr = late_activation_sector_label.astype(np.int)[0,-1,:]
        else:
            late_activation_sector_label_arr = (np.squeeze(late_activation_sector_label >= 0.5)).astype(int)
        # print(late_activation_sector_label_arr.shape)
        late_activation_regions = find_connected_components_binary_1d(late_activation_sector_label_arr, order = 'size')[0]
        if len(late_activation_regions) > 0:
            largest_late_activation_region_center = late_activation_regions[0]['center']
        else:
            largest_late_activation_region_center = 60
        target_sector = largest_late_activation_region_center
    elif target_sector_type == 'center_sector':
        target_sector = 60
    else:
        raise ValueError(f'Unsupported target sector type: {target_sector_type}')
    return target_sector