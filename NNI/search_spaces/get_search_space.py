# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:52:42 2021

@author: remus
"""

def get_default_search_space(exp_type: str):
    if exp_type == 'strainmat_to_TOS':
        search_space = {
            "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
            "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
        }
    elif exp_type == 'multitask_reg_cls':
        search_space = {
            "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
            "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
        }
    else:
        raise ValueError(f'Unsupported experiment type: {exp_type}')
    return search_space

import json
def get_search_space_from_file(filename):
    with open(filename) as file:
        search_space = json.load(file)
    return search_space