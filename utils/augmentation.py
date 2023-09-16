# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:17:33 2021

@author: remus
"""
import numpy as np
import json
def update_augment_argument(aug_args: str, aug_config: list):
    print('UPDATE AUG!')
    print(aug_args)
    aug_args_list = aug_args.split('+')
    existing_aug_methods = [aug_term['method'] for aug_term in aug_config]
    for aug_arg in aug_args_list:
        if len(aug_arg) == 0:
            continue
        else:
            # print('current aug arg: ', aug_arg)
            # print('AAA')
            if aug_arg.startswith('shift-sector'):
                # _, shift_start, shift_end, shift_step = arg.split('-')
                # print('AAA111')
                # print(aug_arg)
                # print(aug_arg.split('='))
                shift_start, shift_end, shift_step = aug_arg.split('=')[1].split('_')
                # print('Shift sector: ', shift_start, shift_end, shift_step)
                if 'shift-sector' in existing_aug_methods:
                    aug_config[existing_aug_methods.index('shift_sector')]['shift_amount'] = np.arange(int(shift_start), int(shift_end), int(shift_step))
                else:
                    aug_config.append({
                        'method': 'shift_sector',
                        'shift_amount': np.arange(int(shift_start), int(shift_end), int(shift_step))
                        })
            elif aug_arg.startswith('mixup'):
                # e.g. mixup_0.5_1000
                mixup_ratio, mixup_amount = aug_arg.split('=')[1].split('_')
                if 'mixup' in existing_aug_methods:
                    aug_config[existing_aug_methods.index('mixup')]['ratio'] = float(mixup_ratio)
                    aug_config[existing_aug_methods.index('mixup')]['amount'] = int(mixup_amount)
                else:
                    aug_config.append({
                        'method': 'mixup',
                        'ratio': float(mixup_ratio),
                        'amount': mixup_amount
                        })
            else:
                raise ValueError(f'Unsupported augmentation command {aug_arg}')
    return aug_config


# def update_augment_argument_json(aug_args_json_str: str, aug_config: list):
def parse_augment_argument_json(aug_args_json_str: str):
    # aug_args_json_str should be a long string containing multiple augmentation commands seperated by "+"
    # For example: 
    # aug_args_json_str = '{"method":"shift_sector", "paras":"-32,32,5", "include_types":"", "exclude_types": 1}+{"method":"mixup", "paras":"0.5,100"}'
    # In each term, the values should be seperated by comma ","
    print('UPDATE AUG!')
    print(aug_args_json_str)
    aug_args_list = aug_args_json_str.split('+')
    aug_config = []
    # existing_aug_methods = [aug_term['method'] for aug_term in aug_config]
    for aug_arg_json_str in aug_args_list:
        aug_arg = json.loads(aug_arg_json_str.strip())
        if len(aug_arg) == 0:
            continue
        else:
            aug_dict = {} 
            aug_dict['method'] = aug_arg['method']
            aug_dict['include_data_conditions'] = aug_arg.get('include_data_conditions').split(',') if aug_arg.get('include_data_conditions', None) is not None else None
            aug_dict['exclude_data_conditions'] = aug_arg.get('exclude_data_conditions').split(',') if aug_arg.get('exclude_data_conditions', None) is not None else None
            # print('current aug arg: ', aug_arg)
            # print('AAA')            
            if aug_arg['method'] == 'shift_sector':
                # _, shift_start, shift_end, shift_step = arg.split('-')
                # print('AAA111')
                # print(aug_arg)
                # print(aug_arg.split('='))
                shift_start, shift_end, shift_step = aug_arg['paras'].split(',')
                # print('Shift sector: ', shift_start, shift_end, shift_step)
                shift_amount = np.arange(int(shift_start), int(shift_end), int(shift_step))
                aug_dict['shift_amount'] = shift_amount                
            elif aug_arg['method'] == 'mixup':
                # e.g. mixup=0.5,1000
                mixup_ratio, mixup_amount = aug_arg['paras'].split(',')
                aug_dict['ratio'] = float(mixup_ratio)
                aug_dict['amount'] = int(mixup_amount)                
            else:
                raise ValueError(f'Unsupported augmentation command {aug_arg}')
            aug_config.append(aug_dict)
    return aug_config

def check_data_condition(datum: dict, conditions: list or None):
    if conditions is None:
        return True
    
    # conditions_list = conditions.split(',')
    check_results = []
    for condition in conditions:        
        if condition == 'has_scar_sector_label':
            # has scar and scar sector label
            meet_current_condition = any([scar_label_key in datum.keys() for scar_label_key in ['scar_sector_label']]) and datum.get('hasScar', 0)!=0
        elif condition == 'has_scar':
            meet_current_condition = datum.get('hasScar', False)
        elif condition == 'no_scar_sector_label':
            # no scar or no scar sector label
            meet_current_condition = any([scar_label_key not in datum.keys() for scar_label_key in ['scar_sector_label']]) or datum.get('hasScar', 0) == 0
            # print(meet_current_condition)
        elif condition == 'no_scar':
            meet_current_condition = not datum.get('hasScar', False)
        else:
            raise ValueError(f'Unrecognized augmentation condition: {condition}')
        check_results.append(meet_current_condition)
    # print(check_results)
    return all(check_results)

def augment(data: list, data_types_to_augment: list, aug_config = [{'method': None}], skip_TOS = False):    
    # print(aug_config)
    data_aug = []
    data_aug_samples = []
    if type(aug_config) is list:
        # If aug_config is a list of configs, execute one by one in order
        for config in aug_config:
            data_aug_curr, data_aug_samples_curr = augment(data, data_types_to_augment, config)
            # print('curr aug: ', len(data_aug_curr))
            data_aug += data_aug_curr
            data_aug_samples.append(data_aug_samples_curr)
        return data_aug, data_aug_samples
    elif type(aug_config) is dict:
        # If aug_config is a config dict, execute augmentation
        # Filter data. Only augment data meeting certain conditions
        curr_aug_include_conditions = aug_config.get('include_data_conditions', None)
        curr_aug_exclude_conditions = aug_config.get('exclude_data_conditions', None)   
        # print(aug_config)
        # print(curr_aug_include_conditions, curr_aug_exclude_conditions)
        # return None
        
        if curr_aug_include_conditions is None and curr_aug_exclude_conditions is None:
            curr_data_to_aug = data
        elif curr_aug_include_conditions is not None and curr_aug_exclude_conditions is None:
            curr_data_to_aug = [datum for datum in data if check_data_condition(datum, curr_aug_include_conditions)]
        elif curr_aug_include_conditions is None and curr_aug_exclude_conditions is not None:
            curr_data_to_aug = [datum for datum in data if not check_data_condition(datum, curr_aug_exclude_conditions)]
        else:
            curr_data_to_aug = [datum for datum in data if check_data_condition(datum, curr_aug_include_conditions) and not check_data_condition(datum, curr_aug_exclude_conditions)]
        # curr_data_to_aug = [datum for datum in data if check_data_condition(datum, curr_aug_include_conditions) and not check_data_condition(datum, curr_aug_exclude_conditions)]
        if len(curr_data_to_aug) == 0:
            print(f'No data meet current include condiation: {curr_aug_include_conditions} and exclude condition: {curr_aug_exclude_conditions}')            
            print([datum['hasScar'] for datum in data])
            data_sample = {}
            data_sample['augmentation'] = aug_config
            return [], data_sample
        if aug_config['method'] in ['shift_sector', 'shift_frame']:
            # print('SHIFT!')
            shift_axis = 'sector' if aug_config['method'] == 'shift_sector' else 'frame'
            shift_amount = aug_config['shift_amount']
            # print('shift_amount, shift_axis', shift_amount, shift_axis)
            # print(type(shift_amount))
            if type(shift_amount) in [list, np.ndarray]:
                for amount in shift_amount:
                    data_aug += shift_data(curr_data_to_aug, data_types_to_augment, shift_axis, amount)
            elif type(shift_amount) is int:
                data_aug += shift_data(curr_data_to_aug, data_types_to_augment, shift_axis, shift_amount)
            
        elif aug_config['method'] == 'mixup':
            # print('MIXUP!')
            mixup_ratio = aug_config.get('ratio', 0.5)
            mixup_amount = aug_config.get('amount', 500)
            data_aug += mixup_data(curr_data_to_aug, data_types_to_augment, mixup_ratio, mixup_amount)
        
        elif aug_config['method'] == 'scale':
            pass
        
        data_sample = data_aug[0] if len(data_aug) >0 else []
        data_sample['augmentation'] = aug_config
        
        print(aug_config, f'{len(curr_data_to_aug)} -> {len(data_aug)}')
        
        return data_aug, data_sample


from utils.data import get_data_category_by_type
def shift_data(data: list, data_types, shift_axis = 'sector', shift_amount = 0):
    if shift_amount == 0:
        return []
    
    # print('SHIFT!')
    data_shifted = []
    for datum in data:
        datum_aug = {}
        for data_type in data_types:
            data_category = get_data_category_by_type(data_type)
            # print(data_category)
            if data_category == 'strainmat':
                # print(data_type)
                if shift_axis == 'sector':
                    datum_aug[data_type] = np.roll(datum[data_type], shift_amount, axis = -2)
                elif shift_axis == 'frame':
                    datum_aug[data_type] = np.roll(datum[data_type], shift_amount, axis = -1)
            
            elif data_category in ['TOS', 'sector_label', 'sector_dist_map']:
                # print(data_type)
                if shift_axis == 'sector':
                    datum_aug[data_type] = np.roll(datum[data_type], shift_amount, axis = -1)
                elif shift_axis == 'frame':
                    datum_aug[data_type] = datum[data_type]
                else:
                    raise ValueError(f'Unsupported type: {data_type}')
            elif data_category in ['data_label']:
                datum_aug[data_type] = datum[data_type]
            elif data_category in ['fit_coefs']:
                pass
            else:
                raise ValueError('Unsupoorted data type: ', data_type)
        
        # print(data_type)
        datum_aug['augmented'] = True
        datum_aug['patient_name'] = datum['patient_name']
        datum_aug['slice_name'] = datum['slice_name']
        datum_aug['patient_slice_name'] = datum['patient_slice_name']
        data_shifted.append(datum_aug)
            
    return data_shifted

def mixup_data(data: list, data_types, mixup_ratio = 0.5, mixup_amount = 500):
    # print('MIXUP!')
    data_mixed = []
    # mixup_ratio = config['mixup']['ratio']
    # mixup_ratio = 0.5
    # mixup_amount = 1000
    # print(mixup_amount)
    for datumIdx in range(int(mixup_amount)):
        datumIdx1, datumIdx2 = np.random.randint(len(data), size=2)
        datum_aug = {}
        for data_type in data_types:
            datum_aug[data_type] = mixup_ratio * data[datumIdx1][data_type] + (1 - mixup_ratio) * data[datumIdx2][data_type]
        datum_aug['augmented'] = True
        datum_aug['patient_name'] = data[datumIdx1]['patient_name'] + '+' + data[datumIdx2]['patient_name']
        datum_aug['slice_name'] = data[datumIdx1]['slice_name'] + '+' + data[datumIdx2]['slice_name']
        datum_aug['patient_slice_name'] = data[datumIdx1]['patient_slice_name'] + '+' +  data[datumIdx2]['patient_slice_name']
        data_mixed.append(datum_aug)

    return data_mixed
