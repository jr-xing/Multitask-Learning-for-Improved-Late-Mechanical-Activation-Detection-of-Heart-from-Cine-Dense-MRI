# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:58:03 2021

@author: remus
"""
import numpy as np
import scipy.io as sio
import random
import warnings
from icecream import ic

def loadmat(filename):
    return sio.loadmat(filename, struct_as_record=False, squeeze_me=True)


import re


def getSliceName(filename):
    filename = filename.replace('\\', '/')
    sliceName = filename.split('/')[-1].split('.mat')[0].replace('_processed', '')
    return sliceName


def getPatientName(filename):
    filename = filename.replace('\\', '/')
    patientName = 'SET' + re.search(r'SET(.*?)/mat', filename).group(1).replace('/', '-')
    return patientName


def train_test_split(config, data: list):
    if config['method'] == 'random_slice':
        training_size = config['paras'].get('training_size', 0.8)
        if training_size > len(data):
            raise ValueError(f'Assigned trainingset size {training_size} is larger than # of data {len(data)}!')

        if training_size < 1:
            training_count = int(np.floor(len(data) * training_size))
        else:
            training_count = training_size

        data_indices = list(range(len(data)))
        random.shuffle(data_indices)
        # data_indices = np.arange(len(data))
        # np.shuffle(data_indices)
        training_data = [data[idx] for idx in data_indices[:training_count]]
        test_data = [data[idx] for idx in data_indices[training_count:]]

        return training_data, test_data

    if config['method'] == 'set_test_by_patient':
        test_patient_names = config['paras'].get('test_patient_names', ['SET01-CT11', 'SET02-CT28', 'SET03-EC21'])
        training_data, test_data = [], []
        for datum in data:
            if datum['patient_name'] in test_patient_names:
                test_data.append(datum)
            else:
                training_data.append(datum)
        return training_data, test_data


def get_data_type_by_category(category: str, data_types, return_index=False):
    # print(data_types)
    # Get data type in data_types within given category
    if category == 'TOS':
        eligible_data_types = ['TOS', 'TOS18_Jerry', 'TOSfullRes_Jerry', 'TOS126', 'TOS18']
    elif category.lower() == 'strainmat':
        eligible_data_types = ['strainMat', 'strainMatSVD', 'strainMatFullResolution', 'strainMatFullResolutionSVD']
    elif category == 'sector_label':
        eligible_data_types = ['late_acti_label', 'strain_curve_type_label', 'scar-AHA-step', 'scar_sector_label', 'late_activation_sector_label']
    elif category == 'sector_dist_map':
        eligible_data_types = ['late_acti_dist_map', 'strain_curve_type_dist_map', 'scar-AHA-distmap', 'scar_sector_distmap']
    elif category == 'sector_value':
        eligible_data_types = ['scar_sector_percentage']
        eligible_data_types += ['TOS', 'TOS18_Jerry', 'TOSfullRes_Jerry', 'TOS126', 'TOS18']
    elif category in ['fit_coefs']:
        eligible_data_types = ['polyfit_coefs']
    # elif category in 'data_label':
    #     eligible_data_types = ['has_scar', 'late_acti_label', 'strain_curve_type_label', 'scar-AHA-step', 'scar_sector_label']
    elif category == 'data_label':
        eligible_data_types = ['has_scar']
    else:
        raise AssertionError(f'Unsupported data category: {category}')

    for data_type_idx, data_type in enumerate(data_types):
        data_type_str = data_type if type(data_type) is str else data_type['type']
        if data_type_str in eligible_data_types:
            data_type = data_type_str
            if return_index:
                return data_type, data_type_idx
            else:
                return data_type
            # return None
    # raise ValueError('Unsupported data type: ', data_type)
    # raise ValueError('data type not found!', category, data_types)
    # raise Warning('data type not found!', category, data_types)
    # warnings.warn(f'data type not found! {category} {data_types}')
    return None


# def get_data_info_by_type(category: str, data_info: list):
def get_data_info_by_category(category: str, data_info: list):    
    data_types = [info['type'] for info in data_info]    
    
    data_type_info = get_data_type_by_category(category, data_types, return_index=True)
    # ic(category)
    if data_type_info is None:
        return None
    else:
        data_type, data_type_idx = data_type_info
        return data_info[data_type_idx]


def get_data_category_by_type(data_type, remove_pred_str=True):
    if remove_pred_str:
        data_type = data_type.replace('_pred', '')
    if data_type in ['TOS', 'TOS18_Jerry', 'TOSfullRes_Jerry', 'activeContourResultFullRes', 'activeContourResult',
                     'pred', 'TOS126', 'TOS18', 'ac_pred']:
        category = 'TOS'
    elif data_type in ['strainMat', 'strainMatSVD', 'strainMatFullResolution', 'strainMatFullResolutionSVD']:
        category = 'strainmat'
    elif data_type in ['late_acti_label', 'strain_curve_type_label', 'scar-AHA-step', 'scar_sector_label', 'late_activation_sector_label']:
        category = 'sector_label'
    elif data_type in ['late_acti_dist_map', 'strain_curve_type_dist_map', 'scar-AHA-distmap', 'scar_sector_distmap']:
        category = 'sector_dist_map'
    elif data_type in ['scar_sector_percentage', 'combined']:
        category = 'sector_value'
    elif data_type in ['polyfit_coefs']:
        category = 'fit_coefs'
    elif data_type in ['has_scar']:
        category = 'data_label'
    else:
        raise ValueError('Unsupported type: ', data_type)
        # warnings.warn(f'Unsupported type: {data_type}')
        # return None
    return category

def get_data_info_by_tag(tag:str, data_info:list):
    # Tags should be unique
    data_info_of_tag = None
    for info in data_info:
        # ic(type(info))
        # ic(info.get('tag'))
        if type(info) is dict and info.get('tag') == tag:
            data_info_of_tag = info
        elif type(info) is str:
            continue
        
    return data_info_of_tag

def generate_strain_curve_types(strainmat, method='default'):
    # strainmat shoudl be [1,1,n_sector, n_frame]
    if method == 'default':
        # sector_max_values_front = np.max(strainmat[:,:,:,:10], axis = -1).flatten()
        sector_max_values = np.max(strainmat, axis=-1).flatten()
        sector_min_values = np.min(strainmat, axis=-1).flatten()
        sector_mean_values = np.mean(strainmat, axis=-1).flatten()

        contract_label = (sector_min_values < -0.10)
        stretch_only_label = (sector_mean_values > 0) * (1 - contract_label)
        other_area_label = (1 - contract_label) * (1 - stretch_only_label)

        strain_curve_type_label = np.vstack([contract_label, stretch_only_label, other_area_label])[None, :]
    else:
        raise ValueError('Unsupported method: ', method)

    return strain_curve_type_label


from scipy import interpolate


def AHA_enhancement_interpolation(slice_enhancement, target_dim=126, method='nearest', threshold=-1):
    if slice_enhancement is None:
        slice_enhancement_interp = np.zeros(target_dim)
    else:
        # Triplicate to keep border continuous
        slice_enhancement_triplicated = np.tile(slice_enhancement, 3)
        dim_each_sector = target_dim / len(slice_enhancement)
        slice_enhancement_triplicated_locs = [dim_each_sector * (0.5 + idx) for idx in
                                              range(len(slice_enhancement_triplicated))]
        f = interpolate.interp1d(slice_enhancement_triplicated_locs, slice_enhancement_triplicated,
                                 fill_value='extrapolate', kind=method)
        slice_enhancement_triplicated_interp_locs = np.arange(target_dim * 3)
        slice_enhancement_triplicated_interp = f(slice_enhancement_triplicated_interp_locs)
        slice_enhancement_interp = slice_enhancement_triplicated_interp[target_dim:target_dim * 2]
        slice_enhancement_interp[slice_enhancement_interp < np.min(slice_enhancement)] = np.min(slice_enhancement)
        if threshold >= 0:
            slice_enhancement_interp[slice_enhancement_interp < threshold] = 0
            slice_enhancement_interp[slice_enhancement_interp >= threshold] = 1
    return slice_enhancement_interp

from utils.scar_utils import remove_sector_label_spkies
def add_classification_label(data: list, data_info, remove_spikes=False, force_onehot=False):
    # UPDATE: for labels, the first class dimension should mean "NO"
    data_types = [info['type'] for info in data_info]

    
    sector_label_data_info = get_data_info_by_category('sector_label', data_info)
    data_label_data_info = get_data_info_by_category('data_label', data_info)
    if sector_label_data_info is None and data_label_data_info is None:
        print('No label type found in: ', data_info)
        return None
    elif sector_label_data_info is not None and data_label_data_info is None:
        label_data_info = sector_label_data_info
    elif sector_label_data_info is None and data_label_data_info is not None:
        label_data_info = data_label_data_info
    else:
        raise ValueError('Should only looking for one label type!')
    
    if label_data_info is None and get_data_info_by_category('sector_dist_map', data_info) is not None:
        label_data_info = get_data_info_by_category('sector_dist_map', data_info)
    
    label_data_type = label_data_info['type']
    # if label_data_type in ['late_acti_label', 'late_acti_dist_map']:
    #     if config.get('generate_method', 'TOS_based') == 'TOS_based':
    #         late_acti_thres = config.get('threshold', 17)
    #         TOS_data_type = get_data_type_by_category('TOS', data_types)
    #         if TOS_data_type is None:
    #             TOS_data_type = 'TOSfullRes_Jerry'
    #         for datum in data:
    #             # datum['lateActiFlag'] = datum[config['data']['outputType']] > 17
    #             lateActiFlag = datum[TOS_data_type] > late_acti_thres
    #             if remove_spikes:
    #                 lateActiFlag = remove_sector_label_spkies(lateActiFlag)
    #             datum['late_acti_label'] = np.concatenate(
    #                 [(lateActiFlag == False)[:, None, :], (lateActiFlag == True)[:, None, :]], axis=1)
    #                 # [(lateActiFlag == True)[:, None, :], (lateActiFlag == False)[:, None, :]], axis=1)
    #             # datum['lateActiFlag'] = np.concatenate([(lateActiFlag==False)[:, None, None, :], (lateActiFlag==True)[:,None, None, :]], axis=-1)
    #             # output shape: (1, 2, n_sector)
    if label_data_type in ['late_activation_sector_label']:
        if label_data_info.get('generate_method', 'TOS_based') == 'TOS_based':
            late_acti_thres = label_data_info.get('threshold', 17+1e-3)
            TOS_data_type = get_data_type_by_category('TOS', data_types)
            if TOS_data_type is None:
                TOS_data_type = 'TOS126'
            for datum in data:
                # datum['lateActiFlag'] = datum[config['data']['outputType']] > 17
                lateActiFlag = np.squeeze(datum[TOS_data_type] > late_acti_thres)
                if remove_spikes:
                    lateActiFlag = remove_sector_label_spkies(lateActiFlag)
                if force_onehot:
                    datum[label_data_type] = np.concatenate(
                        [(lateActiFlag == False)[None, None, :], (lateActiFlag == True)[None, None, :]], axis=1).astype(float)
                else:
                    datum[label_data_type] = lateActiFlag.astype(float)[None, :]
                    # [(lateActiFlag == True)[:, None, :], (lateActiFlag == False)[:, None, :]], axis=1)
                # datum['lateActiFlag'] = np.concatenate([(lateActiFlag==False)[:, None, None, :], (lateActiFlag==True)[:,None, None, :]], axis=-1)
                # output shape: (1, 2, n_sector)
    elif label_data_type in ['strain_curve_type_label', 'strain_curve_type_dist_map']:
        for datum in data:
            strainmat_data_type = get_data_type_by_category('strainmat', data_types)
            strainmat = datum[strainmat_data_type]
            datum['strain_curve_type_label'] = generate_strain_curve_types(strainmat, method='default')
    elif label_data_type in ['scar-AHA-step']:
        # Add full resolution scar AHA annotation
        for datum in data:
            scar_AHA_raw = datum['scar_AHA_raw']
            scar_AHA_interpolated = AHA_enhancement_interpolation(scar_AHA_raw, target_dim=126, method='nearest',
                                                                  threshold=label_data_info['config']['scar_region_threshold'])
            datum['scar-AHA-step'] = np.zeros((1, 2, len(scar_AHA_interpolated)))
            datum['scar-AHA-step'][:, 0, :] = 1 - scar_AHA_interpolated
            datum['scar-AHA-step'][:, 1, :] = scar_AHA_interpolated
    elif label_data_type in ['scar-AHA-distmap']:
        pass
    elif label_data_type in ['scar_sector_label']:
        
        strainmat_data_type = get_data_type_by_category('strainMat', data[0].keys())
        N_sectors = data[0][strainmat_data_type].shape[-2]
        for datum in data:
            if force_onehot:
                datum['scar_sector_label'] = np.zeros((1, 2, N_sectors))
                if 'scar_sector_percentage' not in datum.keys():
                    datum['scar_sector_percentage'] = np.zeros(N_sectors)
                    datum['scar_sector_label'][0, 0, :] = 1                
                else:
                    scar_exists_each_sector = datum['scar_sector_percentage'] >= 0.5
                    if remove_spikes:        
                        # print('remove_spikes!', datum['patient_slice_name'])
                        scar_exists_each_sector = remove_sector_label_spkies(scar_exists_each_sector)
                    datum['scar_sector_label'][0, 0, :] = 1 - scar_exists_each_sector
                    datum['scar_sector_label'][0, 1, :] = scar_exists_each_sector
            else:
                datum['scar_sector_label'] = np.zeros((1, N_sectors))
                if 'scar_sector_percentage' not in datum.keys():
                    datum['scar_sector_percentage'] = np.zeros(N_sectors)
                    datum['scar_sector_label'][0, :] = 0
                else:
                    scar_exists_each_sector = datum['scar_sector_percentage'] >= 0.5
                    if remove_spikes:        
                        # print('remove_spikes!', datum['patient_slice_name'])
                        scar_exists_each_sector = remove_sector_label_spkies(scar_exists_each_sector)
                    datum['scar_sector_label'][0, :] = scar_exists_each_sector
    elif label_data_type in ['has_scar']:
        for datum in data:
            if force_onehot:
                datum['has_scar'] = np.zeros([1,2,1])
                if datum.get('hasScar', False):
                    datum['has_scar'][0, 1, 0] = 1
                else:                
                    datum['has_scar'][0, 0, 0] = 1
            else:
                datum['has_scar'] = np.zeros([1,1])
                datum['has_scar'][0, 0] = int(datum.get('hasScar', False))
    else:
        raise ValueError('Unsupported class type: ', label_data_type)


def generate_distance_map(label, N_sectors=0):
    # label: should be 1d array
    if np.all((label == 0)):
        return -np.ones(len(label)) * (N_sectors // 2)
    label_circ = np.hstack([label.flatten()] * 3)
    
    # print(len(label))

    label_boundary_flag = np.roll(label_circ, 1) - label_circ
    label_boundary_locs = np.where(np.abs(label_boundary_flag) > 0)[0]
    label_distance_raw = np.zeros(len(label_circ))
    for loc in range(len(label_distance_raw)):
        distance_raw = np.min(np.abs(loc - label_boundary_locs))
        label_distance_raw[loc] = distance_raw

    label_distance = np.zeros(len(label_circ))
    label_distance[label_circ > 0.5] = label_distance_raw[label_circ > 0.5]
    label_distance[label_circ < 0.5] = -label_distance_raw[label_circ < 0.5]
    
    # print(len(label))
    label_distance = label_distance[len(label):2 * len(label)]

    return label_distance

# def add_distancce_map(data: list, config, data_types):
def add_distance_map(data: list, data_info, remove_spikes=False):
    # print(data_info)
    data_types = [info['type'] for info in data_info]
    # if config['class_type'] == 'late_acti_dist_map':
    if 'late_acti_dist_map' in data_types:
        # n_sectors = data[0]['late_acti_label'].shape[-1]
        for datum in data:
            late_acti_flag = datum['late_acti_label'][:, -1, :].flatten().astype(np.float)            
            late_acti_dist = generate_distance_map(late_acti_flag)

            datum['late_acti_dist_map'] = late_acti_dist[None, :]
    elif 'strain_curve_type_dist_map' in data_types:
        for datum in data:
            strain_curve_type_label = datum['strain_curve_type_label']
            label_dist_maps = []
            for label_class in range(1, strain_curve_type_label.shape[1]):
                label = strain_curve_type_label[0, label_class, :]
                label_dist_map = generate_distance_map(label)[None, None, :]
                label_dist_maps.append(label_dist_map)

            datum['strain_curve_type_dist_map'] = np.concatenate(label_dist_maps, axis=1)
            # strain_curve_type_dist = generate_distance_map(strain_curve_type_label)
            # datum[]
    elif 'scar-AHA-distmap' in data_types:
        for datum in data:
            scar_AHA_raw = datum['scar_AHA_raw']
            scar_AHA_interpolated = AHA_enhancement_interpolation(scar_AHA_raw, target_dim=126, method='cubic')
            datum['scar-AHA-distmap'] = scar_AHA_interpolated[None, None, :]
    elif 'scar_sector_distmap' in data_types:
        strainmat_data_type = get_data_type_by_category('strainMat', data[0].keys())
        N_sectors = data[0][strainmat_data_type].shape[-2]
        for datum in data:
            # datum['scar_sector_label'] should be (1, 2, N_sectors) array
            # where datum['scar_sector_label'][:, 0, :] is non-scar label and datum['scar_sector_label'][0,1,:] is scar label
            if 'scar_sector_label' not in datum.keys():
                datum['scar_sector_label'] = np.zeros((1, 2, N_sectors))
                if 'scar_sector_percentage' not in datum.keys():
                    datum['scar_sector_label'][0, 0, :] = 1
                else:
                    scar_exists_each_sector = datum['scar_sector_percentage'] >= 0.5
                    if remove_spikes:
                        scar_exists_each_sector = remove_sector_label_spkies(scar_exists_each_sector)
                    datum['scar_sector_label'][0, 0, :] = 1 - scar_exists_each_sector
                    datum['scar_sector_label'][0, 1, :] = scar_exists_each_sector
            
            scar_sector_label = datum['scar_sector_label'][0, 1, :]
            datum['scar_sector_distmap'] = generate_distance_map(scar_sector_label, N_sectors)[None, None, :]            
    elif any([TOS_type in data_types for TOS_type in ['TOS18', 'TOS126']]):
        pass
    else:
        raise ValueError('Unsupported type: ', data_types)


def add_polyfit_coefficient(data: list, data_types, degree=15):
    # https://towardsdatascience.com/polynomial-regression-with-scikit-learn-what-you-should-know-bed9d3296f2
    TOS_type = get_data_type_by_category('TOS', data_types)
    if TOS_type is None:
        TOS_type = 'TOSfullRes_Jerry'
    for datum in data:
        TOS = datum[TOS_type].flatten()
        xs = np.arange(TOS.size)
        coefs = np.polyfit(xs, TOS, degree)
        # coefs[coefs < 1e-5] = 0
        # approx = np.polyval(coefs, xs)
        # datum['polyfit_coefs'] = np.log(coefs)[None,:]

        coefs_mags = np.floor(np.log10(np.abs(coefs)))
        coefs_vals = coefs / 10 ** coefs_mags

        coefs_sep = np.zeros(len(coefs) * 2)
        coefs_sep[0::2] = coefs_vals
        coefs_sep[1::2] = coefs_mags
        # coefs_sep[]

        # datum['polyfit_coefs'] = coefs[None,:]
        datum['polyfit_coefs'] = coefs_sep[None, :]

def report_data_info(data: list, 
                     report_total_number=True,
                     report_scar_info=True):
    if report_total_number:
        print('Total number: ', len(data))
    if report_scar_info:
        data_without_scar_num = len([datum for datum in data if datum.get('hasScar', False) == False])
        data_with_scar_num = len([datum for datum in data if datum.get('hasScar', False) == True])
        print(f'{data_without_scar_num} don\'t contain scar, {data_with_scar_num} have')