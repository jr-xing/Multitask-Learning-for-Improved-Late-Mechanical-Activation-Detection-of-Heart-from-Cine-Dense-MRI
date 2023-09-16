# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:53:27 2021

@author: Jerry Xing
"""

def get_default_search_space(exp_type: str):
    if exp_type in ['strainmat_to_TOS', 'multitask_reg_cls']:
        search_space = {
            "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
            "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
        }
    else:
        raise ValueError(f'Unsupported experiment type: {exp_type}')
    return search_space

def get_default_config(exp_type: str):
    if exp_type in ['strainmat_to_TOS', 'multitask_reg_cls', 'multitask-reg-cls', 'multitask_reg_clsDistMap', 'cls', 'reg']:
        config = {
            'authorName': 'default',
            'experimentName': 'example_mnist_pytorch',
            'trialConcurrency': 1,
            'maxExecDuration': '999d',
            'maxTrialNum': 10,
            #choice: local, remote, pai
            'trainingServicePlatform': 'local',
            'searchSpacePath': 'search_space.json',
            #choice: true, false
            'useAnnotation': False,
            'tuner':
            {
              #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
              #SMAC (SMAC should be installed through nnictl)
              'builtinTunerName': 'TPE',
              'classArgs':
                #choice: maximize, minimize
                {'optimize_mode': 'minimize'},            
            },
            'trial':
            {
                'command': 'python mainV6_nni_run_exp.py',
                'codeDir': '.',
                'gpuNum': 1
            },
            'localConfig':
                {'useActiveGpu': True}
        }
    return config

import yaml
def get_search_space_from_file(filename: str):
    with open(filename) as file:
        search_space = yaml.load(file)
    
    return search_space 