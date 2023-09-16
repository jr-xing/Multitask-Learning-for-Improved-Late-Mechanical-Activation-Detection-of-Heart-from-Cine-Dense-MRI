# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:00:57 2021

@author: Jerry Xing
"""

import numpy as np
def find_connected_components_binary_1d(arr_bin, order='index'):
    # return: an array, 0 -> background, 1,2,3,... different region labels
    arr_bin_padded = np.concatenate((np.zeros(1), arr_bin))
    region_label = np.zeros_like(arr_bin)
    curr_region_idx = 0
    is_inside_region = False
    for idx in range(len(arr_bin)):
        # up stair -> new region
        if arr_bin_padded[idx] == 0 and arr_bin_padded[idx + 1] == 1:
            is_inside_region = True
            curr_region_idx += 1
                
        # down stair -> end of region
        if arr_bin_padded[idx] == 1 and arr_bin_padded[idx + 1] == 0:
            is_inside_region = False        
        
        if is_inside_region:
            region_label[idx] = curr_region_idx
    
    regions = []
    for region_idx in range(1, curr_region_idx + 1):
        region_center = int(np.mean(np.where(region_label == region_idx)))
        region_length = np.sum(region_label == region_idx)
        regions.append({
            'center': region_center,
            'length': region_length
            })
    if order == 'size':
        region_lengths = [region['length'] for region in regions]
        regions = [regions[idx] for idx in np.argsort(region_lengths)[::-1]]
    
    return regions, region_label


def has_spike(binary_array, len_thres=5):
    if np.sum(binary_array) <= 1:
        return False
    connected_regions, _ = find_connected_components_binary_1d(binary_array, order='size')
    minimal_connected_region_len = connected_regions[-1]['length']
    if minimal_connected_region_len <= len_thres:
        return True
    else:
        return False

def has_gap(binary_array, len_thres=5):
    return has_spike(1 - binary_array, len_thres)

from scipy.ndimage import binary_closing
def remove_sector_label_spkies(sector_binary_label, connect_len=20, check_has_spike_or_gap=True):
    if np.ndim(np.squeeze(sector_binary_label)) > 1:
        raise ValueError('Only support 1D array, but got data with shape ', sector_binary_label.shape)
        
    sector_binary_label_int = np.squeeze(sector_binary_label).astype(int)
    # print(sector_binary_label_int)
    if check_has_spike_or_gap:
        if not (has_spike(sector_binary_label_int) or has_gap(sector_binary_label_int)):
            # if no spike nor gap, return original array
            # print('No spike nor gap!')
            return sector_binary_label
    # print(connect_len)
    sector_label_int_repeated = np.tile(sector_binary_label_int, 3)
    sector_label_int_repeated_closed = binary_closing(sector_label_int_repeated, structure=np.ones((connect_len)))
    # print(sector_label_int_repeated_closed[len(sector_binary_label):2*len(sector_binary_label)].astype(float))
    return sector_label_int_repeated_closed[len(sector_binary_label):2*len(sector_binary_label)].astype(float)

if __name__ == '__main__':
    arr_float = np.array([0.06720928, 0.05567385, 0.01683778, 0.        , 0.        ,
       0.        , 0.01444707, 0.03438657, 0.04572006, 0.06063397,
       0.09380722, 0.12663361, 0.06253705, 0.02470547, 0.01385187,
       0.0168075 , 0.0228027 , 0.02326755, 0.01844862, 0.01072838,
       0.00685119, 0.00882822, 0.10571809, 0.50890665, 0.81873536,
       0.99646353, 1.        , 1.        , 1.        , 1.        ,
       0.85006615, 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 0.79407982, 0.76392822,
       1.        , 0.83103726, 0.77070112, 0.90187895, 0.713213  ,
       0.5749466 , 0.49728279, 0.37683981, 0.19761897, 0.15851356,
       0.09908356, 0.07868099, 0.06306678, 0.01934582, 0.01280275,
       0.05982818, 0.08568501, 0.04796073, 0.01151019, 0.00392487,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.00880832,
       0.01937589, 0.03698034, 0.03849519, 0.02257884, 0.01211493,
       0.00155185, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.00215863,
       0.02509387])
    find_connected_components_binary_1d((arr_float > 0.1).astype(np.int))
