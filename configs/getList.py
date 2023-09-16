# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 08:22:12 2020

@author: remus
"""

def get_list(name, levels = ['2','3','4','5']):
    if name == 'problematic_AC_18':
        data_list = [
            {'name':'SET01-CT01-Mid4', 'level':'3'},
            {'name':'SET01-CT11-SL1', 'level':'3'},
            {'name':'SET01-CT11-SL2', 'level':'3'},
            {'name':'SET01-CT11-SL3', 'level':'3'},
            {'name':'SET01-CT11-SL4', 'level':'3'},
            {'name':'SET01-CT11-SL5', 'level':'3'},
            {'name':'SET01-CT11-SL6', 'level':'3'},
            {'name':'SET01-CT11-SL7', 'level':'3'},
            {'name':'SET01-CT15-SL1', 'level':'3'},
            {'name':'SET01-CT15-SL4', 'level':'3'},
            {'name':'SET01-CT20-SL_86', 'level':'3'},
            {'name':'SET01-CT26-SL7', 'level':'3'},
            {'name':'SET02-CT28-SL1', 'level':'3'},
            {'name':'SET02-CT28-SL6', 'level':'2'},
            {'name':'SET02-Ct36-SL2', 'level':'3'},
            {'name':'SET02-Ct36-SL3', 'level':'3'},
            {'name':'SET02-Ct36-SL4', 'level':'3'},
            {'name':'SET02-CT39-SL2', 'level':'3'},
            {'name':'SET02-CT39-SL5', 'level':'3'},
            {'name':'SET02-EC10-SA1', 'level':'2'},
            {'name':'SET03-EC10-SA3', 'level':'3'},
            {'name':'SET03-EC221-SL6', 'level':'3'},
            ]

    elif name == 'problematic_AC_126':
        data_list = [
            {'name':'SET01-CT01-Mid4', 'level':'3'},
            {'name':'SET01-CT11-SL1', 'level':'3'},
            {'name':'SET01-CT11-SL2', 'level':'3'},
            # {'name':'SET01-CT11-SL3', 'level':'3'},
            {'name':'SET01-CT11-SL4', 'level':'3'},
            {'name':'SET01-CT11-SL5', 'level':'3'},
            {'name':'SET01-CT11-SL6', 'level':'3'},
            {'name':'SET01-CT11-SL7', 'level':'3'},
            {'name':'SET01-CT15-SL1', 'level':'3'},
            {'name':'SET01-CT15-SL4', 'level':'3'},
            {'name':'SET01-CT20-SL_86', 'level':'3'},
            {'name':'SET01-CT26-SL7', 'level':'3'},
            {'name':'SET02-CT28-SL1', 'level':'3'},
            {'name':'SET02-CT28-SL6', 'level':'2'},
            {'name':'SET02-Ct36-SL2', 'level':'3'},
            {'name':'SET02-Ct36-SL3', 'level':'3'},
            {'name':'SET02-Ct36-SL4', 'level':'3'},
            {'name':'SET02-CT39-SL2', 'level':'3'},
            {'name':'SET02-CT39-SL5', 'level':'3'},
            {'name':'SET02-EC10-SA1', 'level':'2'},
            {'name':'SET03-EC10-SA3', 'level':'3'},
            {'name':'SET03-EC221-SL6', 'level':'3'},
            ]
    elif name == 'problematic_GT':
        data_list = [
            {'name':'SET01-CT01-base', 'level':'2'},
            {'name':'SET01-CT01-Mid4', 'level':'2'},
            {'name':'SET01-CT11-SL1', 'level':'5'},
            {'name':'SET01-CT11-SL2', 'level':'5'},
            {'name':'SET01-CT11-SL3', 'level':'4'},
            {'name':'SET01-CT11-SL4', 'level':'3'},
            {'name':'SET01-CT11-SL5', 'level':'2'},
            {'name':'SET01-CT11-SL7', 'level':'4'},
            {'name':'SET01-CT11-SL8', 'level':'4'},
            {'name':'SET01-CT15-SL2', 'level':'3'},
            {'name':'SET01-CT15-SL4', 'level':'3'},
            {'name':'SET01-CT15-SL5', 'level':'2'},
            {'name':'SET01-CT15-SL6', 'level':'3'},
            {'name':'SET01-CT19-SL1', 'level':'3'},
            {'name':'SET01-CT19-SL2', 'level':'3'},
            {'name':'SET01-CT19-SL3', 'level':'2'},
            {'name':'SET01-CT19-SL4', 'level':'2'},
            {'name':'SET01-CT20-SL_110', 'level':'3'},
            {'name':'SET01-CT20-SL_118', 'level':'3'},
            {'name':'SET01-CT20-SL_126', 'level':'2'},
            {'name':'SET01-CT20-SL_78', 'level':'3'},
            {'name':'SET01-CT20-SL_86', 'level':'3'},
            {'name':'SET01-CT20-SL_94', 'level':'2'},
            {'name':'SET02-EC02-Mid 1', 'level':'5'},
            {'name':'SET02-EC02-Mid 3', 'level':'4'},
            {'name':'SET02-EC02-Mid 4', 'level':'3'},
            {'name':'SET02-EC03-Base', 'level':'2'},
            {'name':'SET02-EC03-mid2', 'level':'2'},
            {'name':'SET02-CT26-SL9', 'level':'2'},
            {'name':'SET02-CT26_SL8', 'level':'3'},
            {'name':'SET02-CT28-SL1', 'level':'3'},
            {'name':'SET02-CT28-SL2', 'level':'3'},
            {'name':'SET02-CT28-SL3', 'level':'3'},
            {'name':'SET02-CT28-SL4', 'level':'3'},
            {'name':'SET02-CT28-SL5', 'level':'2'},
            {'name':'SET02-CT33-SL2', 'level':'2'},
            {'name':'SET02-CT33-SL3', 'level':'2'},
            {'name':'SET02-CT33-SL4', 'level':'2'},
            {'name':'SET02-CT33-SL5', 'level':'2'},
            {'name':'SET02-CT36-SL7', 'level':'3'},
            {'name':'SET02-CT39-SL2', 'level':'3'},
            {'name':'SET02-CT39-SL3', 'level':'2'},
            {'name':'SET02-CT39-SL6', 'level':'3'},
            {'name':'SET02-CT39-SL7', 'level':'2'},
            {'name':'SET03-EC10-SA3', 'level':'3'},
            {'name':'SET03-EC10-SA5', 'level':'3'},
            {'name':'SET03-EC21-SL6', 'level':'3'},
            ]
    elif name == 'problematic_GT_2':
        data_list = ['SET01-CT15-SL1', 
                     'SET01-CT15-SL4',
                     'SET01-CT11-SL1',
                     'SET01-CT11-SL2',
                     'SET01-CT11-SL3',
                     'SET01-CT11-SL4']
        data_list += [f'SET01-CT12_s-SL{idx}' for idx in range(1,9)]
    if type(data_list) is list:
        return data_list
    else:
        return [item for item in data_list if item['level'] in levels]
    