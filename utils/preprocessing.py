# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:50:44 2021

@author: remus
"""
import numpy as np
from skimage.transform import resize
from utils.data import get_data_category_by_type
def remove_last_frames(data: list, data_types: list, n_frames_to_remove=0):
    if n_frames_to_remove == 0:
        return None    
    for data_type in data_types:
        if data_type in ['strainMat', 'strainMatSVD', 'strainMatFullResolution', 'strainMatFullResolutionSVD']:            
            print('Remove last ', n_frames_to_remove, ' frames')
            for datum in data:
                datum[data_type] = datum[data_type][..., :-n_frames_to_remove]                                    
        elif get_data_category_by_type(data_type) in ['TOS', 'sector_label', 'sector_dist_map', 'fit_coefs', 'sector_value', 'data_label']:
            pass
        else:
            raise ValueError('Unsupported data type: ', data_type)

def unify_n_frame(data: list, data_types: list, n_frames_target=None, method='zero_padding'):
    for data_type in data_types:
        if data_type in ['strainMat', 'strainMatSVD', 'strainMatFullResolution', 'strainMatFullResolutionSVD']:
            if n_frames_target is None:
                n_frames_target = np.max([datum[data_type].shape[-1] for datum in data])
            elif n_frames_target == 'power_of_2':
                n_frames_max = np.max([datum[data_type].shape[-1] for datum in data])
                for power in range(10):
                    if 2 ** power >= n_frames_max:
                        n_frames_target = 2 ** power
                        break
            print('Unify # of frame to ', n_frames_target)
            for datum in data:
                if method == 'zero_padding':
                    n_frames_datum = datum[data_type].shape[-1]
                    if n_frames_datum == n_frames_target:
                        pass
                    elif n_frames_datum < n_frames_target:
                        padding_shape = list(datum[data_type].shape[:-1]) + [n_frames_target - n_frames_datum]
                        datum[data_type] = np.concatenate((datum[data_type], np.zeros(padding_shape)), axis=-1)
                    elif n_frames_datum > n_frames_target:
                        datum[data_type] = datum[data_type].take(indices=np.arange(n_frames_target, axis=-1))
                elif method == 'resize':
                    raise ValueError('Unsupported yet')
        elif get_data_category_by_type(data_type) in ['TOS', 'sector_label', 'sector_dist_map', 'fit_coefs', 'sector_value', 'data_label']:
            pass
        else:
            raise ValueError('Unsupported data type: ', data_type)



def unify_n_sector(data: list, data_types_ori: list, n_sectors_target=None, method='copy_boundary'):
    # Add 
    # print('scar_sector_distmap' in data_types)
    data_types = data_types_ori.copy()
    if ('scar_sector_label' in data_types or 'scar_sector_distmap' in data_types) and 'scar_sector_percentage' not in data_types:
        data_types.append('scar_sector_percentage')
        # print('append scar perc')
    
    for data_type in data_types:
        data_category = get_data_category_by_type(data_type)
        if data_category == 'strainmat':
            if n_sectors_target is None:
                n_sectors_target = np.max([datum[data_type].shape[-2] for datum in data])
            elif n_sectors_target == 'power_of_2':
                n_sectors_max = np.max([datum[data_type].shape[-2] for datum in data])
                for power in range(10):
                    if 2 ** power >= n_sectors_max:
                        n_sectors_target = 2 ** power
                        break
            print('Unify # of sector to ', n_sectors_target)
            # print(n_sectors_target)
            for datum in data:
                n_sectors_datum = datum[data_type].shape[-2]
                if n_sectors_target == n_sectors_datum:
                    pass
                elif n_sectors_datum < n_sectors_target:
                    padding_shape = list(datum[data_type].shape)
                    padding_shape[-2] = n_sectors_target - n_sectors_datum
                    if method == 'zero_padding':
                        datum[data_type] = np.concatenate((datum[data_type], np.zeros(padding_shape)), axis=-2)
                    elif method == 'copy_boundary':
                        copied_boundary = np.repeat(datum[data_type].take([0], axis=-2),
                                                    n_sectors_target - n_sectors_datum, axis=-2)
                        datum[data_type] = np.concatenate((copied_boundary, datum[data_type]), axis=-2)
                elif n_sectors_datum > n_sectors_target:
                    datum[data_type] = datum[data_type].take(indices=np.arange(n_sectors_target), axis=-2)

        elif data_category in ['TOS', 'sector_label', 'sector_value']:
            if n_sectors_target is None:
                n_sectors_target = np.max([datum[data_type].shape[-1] for datum in data])
            elif n_sectors_target == 'power_of_2':
                n_sectors_max = np.max([datum[data_type].shape[-1] for datum in data])
                for power in range(10):
                    if 2 ** power >= n_sectors_max:
                        n_sectors_target = 2 ** power
                        break
                        
            for datum in data:                                        
                if data_type not in datum.keys():
                    continue
                n_sectors_datum = datum[data_type].shape[-1]                
                if n_sectors_target == n_sectors_datum:
                    pass
                elif n_sectors_datum < n_sectors_target:
                    padding_shape = list(datum[data_type].shape)
                    padding_shape[-1] = n_sectors_target - n_sectors_datum
                    if method == 'zero_padding':
                        datum[data_type] = np.concatenate((datum[data_type], np.zeros(padding_shape)), axis=-2)
                    elif method == 'copy_boundary':
                        # copied_boundary = np.repeat(datum[data_type].take([0], axis=-1), n_sectors_target - n_sectors_datum, axis=-1)
                        # datum[data_type] = np.concatenate((copied_boundary, datum[data_type]), axis = -1)
                        copied_boundary = np.repeat(datum[data_type].take([-1], axis=-1),
                                                    n_sectors_target - n_sectors_datum, axis=-1)
                        datum[data_type] = np.concatenate((datum[data_type], copied_boundary), axis=-1)
                elif n_sectors_datum > n_sectors_target:
                    datum[data_type] = datum[data_type].take(indices=np.arange(n_sectors_target), axis=-1)
        # elif data_category in ['sector_label', 'sector_dist_map', 'fit_coefs']:
        elif data_category in ['fit_coefs', 'sector_dist_map', 'data_label']:
            pass
        else:
            raise ValueError('Unsupported data category: ', data_category)
