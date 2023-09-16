# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:50:48 2021

@author: Jerry Xing
"""

# %% 1. Load Search Space
# from NNI.search_spaces.get_search_space import get_default_search_space
from pathlib import Path
import NNI.utils as nni_utils
import sys
# exp_type = 'strainmat_to_TOS'
# exp_type = 'multitask_reg_cls'
# exp_type = 'multitask_reg_clsDistMap'
# search_space = nni_utils.get_default_search_space(exp_type = exp_type)
# search_space = nni_utils.get_search_space_from_file('2021-02-17-multitask-wide-search')
import yaml

# search_space_filename = './NNI/search_spaces/2021-03-07-multitask.yml'
# search_space_filename = './NNI/search_spaces/2021-03-10-multitask-strain-type-label-scar.yml'
if len(sys.argv) <= 1:
    print('sys.argv: ', sys.argv)
    raise ValueError('Need to specify search space ymal file!')
else:
    search_space_filename = sys.argv[1]
if len(sys.argv) == 3:
    port = int(sys.argv[2])
else:
    port = 8090
# search_space_filename = './NNI/search_spaces/2021-06-10-multitask-LBBB-with-scar.yml'
exp_name = search_space_filename.split('/')[-1].split('.yml')[0]
with open(search_space_filename) as file:
    search_space = yaml.load(file, Loader=yaml.FullLoader)
exp_type = search_space['exp_type']['_value'][0]
# %% 2. Create folder for experiment
# import os
from utils.io import create_folder
from datetime import date

current_machine = 'CS-Server'
if current_machine == 'AAR8-Windows':
    parent_path_to_save_exp = 'D:\\Research\\Cardiac\\Experiment_Results\\'
elif current_machine == 'CS-Server':
    # parent_path_to_save_exp = '../../experiment_results/'
    parent_path_to_save_exp = '../../../cardiac-exp-results/NNI_test/'
elif current_machine == 'Tower328':
    parent_path_to_save_exp = '/home/jrxing/WorkSpace/Research/Cardiac/experiment_results/'

# path_to_exp = parent_path_to_save_exp + exp_type + '-' + date.today().strftime('%Y-%m-%d')
# path_to_exp = parent_path_to_save_exp + date.today().strftime('%Y-%m-%d') + '-' + exp_type
path_to_exp = str(Path(parent_path_to_save_exp, exp_name))
path_to_exp = create_folder(path_to_exp, recursive=False, action_when_exist='add index')
create_folder(str(Path(path_to_exp, 'training_results')), recursive=False, action_when_exist='pass')
create_folder(str(Path(path_to_exp, 'test_results')), recursive=False, action_when_exist='pass')
create_folder(str(Path(path_to_exp, 'training_results_CAM')), recursive=False, action_when_exist='pass')
create_folder(str(Path(path_to_exp, 'test_results_CAM')), recursive=False, action_when_exist='pass')
create_folder(str(Path(path_to_exp, 'networks')), recursive=False, action_when_exist='pass')
create_folder(str(Path(path_to_exp, 'configs')), recursive=False, action_when_exist='pass')

# %% 3. Create NNI experiment config files
import json

search_space_filename = str(Path(path_to_exp, 'search_space.json'))
with open(search_space_filename, 'w') as file:
    json.dump(search_space, file, indent=4)

import yaml
# from pathlib import Path
import os

config = nni_utils.get_default_config(exp_type)
config['authorName'] = 'Jerry'
config['experimentName'] = exp_name
config['trialConcurrency'] = 4
config['maxTrialNum'] = 50
config['searchSpacePath'] = search_space_filename
config['tuner']['builtinTunerName'] = 'TPE'
config['tuner']['classArgs']['optimize_mode'] = 'minimize'
# config['trial']['command'] = 'python run_exp_func.py'
config['trial']['command'] = 'python run_exp_script.py'
# config['trial']['codeDir'] = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\strainmat_to_tos\\codeV6\\'
config['trial']['codeDir'] = os.path.dirname(os.path.realpath(__file__))
config['trial']['gpuNum'] = 1
# config['localConfig']['useActiveGpu'] = True
config_filename = str(Path(path_to_exp, 'config.yml'))
with open(config_filename, 'w') as file:
    yaml.dump(config, file)

exp_info = config.copy()
exp_info['exp_parent_path'] = path_to_exp
from pathlib import Path, PurePath

# with open(str(Path.home()) + f'\\nni-config-{date.today().strftime("%Y-%m-%d")}.yml', 'w') as file:
exp_yml_filename = str(PurePath(Path.home(), f'nni-config-{date.today().strftime("%Y-%m")}.yml'))
with open(exp_yml_filename, 'w') as file:
    yaml.dump(exp_info, file)

exp_trial_performance_log_filename = PurePath(path_to_exp, 'trial_performance_log.yml')
with open(exp_trial_performance_log_filename, 'w') as file:
    # yaml.dump({''}, file)
    pass

# %% 4. Run
# out = subprocess.run(['python', '--version'], stdout=PIPE, stderr=PIPE, shell=True)
# print(out.stdout)
# print(out.stderr)
# import os
# os.system(f'conda activate automl & nnictl create --config {config_filename}')
# port = '8090'
os.system(f'nnictl create --config {config_filename} --port={port}')
