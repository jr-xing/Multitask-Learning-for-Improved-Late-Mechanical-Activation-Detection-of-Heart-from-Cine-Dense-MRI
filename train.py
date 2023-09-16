# %% 1. Import libraries
import logging
from configs.getconfig import get_default_config, modify_config, update_iotype_argument
from utils.io import create_folder

import nni
import argparse
import torch
import yaml, json
from pathlib import Path, PurePath
from datetime import date
import numpy as np
from utils.io import load_data_from_table
from icecream import ic
import pprint

# %% 2. Load config

_logger = logging.getLogger('mnist_example')
_logger.setLevel(logging.INFO)
from utils.augmentation import update_augment_argument, parse_augment_argument_json

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument("--inputType", type=str,
    #                     default='strainMat', help="data directory")
    parser.add_argument('--n_conv_layers', type=int, default=3, metavar='NCL',
                        help='n_conv_layers (default: 3)')
    parser.add_argument('--n_conv_channels', type=int, default=3, metavar='NCC',
                        help='n_conv_channels (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--augmentatuon', type=str, default='', metavar='AUG',
                        help='aug (default: "")')
    parser.add_argument('--aug_more_on_data_with_scar', type=bool, default=False, metavar='AS',
                        help='aug_more_on_data_with_scar (default: False)')
    parser.add_argument('--scar_free', type=str, default="True", metavar='SCAR',
                        help='Avoid Data with scar or not (default: "True")')
    parser.add_argument('--input_types', type=str, default="strainMat", metavar='IN',
                        help='Input types (Split by "+")')
    parser.add_argument('--output_types', type=str, default="TOS", metavar='OUT',
                        help='Output types (Split by "+")')

    for network_para in ['joint_n_conv_layers', 'joint_n_conv_channels', 'joint_conv_size',
                         'reg_n_conv_layers', 'reg_n_conv_channels', 'reg_conv_size', 'reg_n_linear_layers',
                         'cls_n_conv_layers', 'cls_n_conv_channels', 'cls_conv_size', 'cls_n_linear_layers']:
        parser.add_argument(f'--{network_para}', type=int, default=-1)

    parser.add_argument('--regularize_weight', type=float, default=0, metavar='WREGU',
                        help='Regularize_weight (default: 0)')
    parser.add_argument('--cls_weight', type=float, default=1e1, metavar='CLSW',
                        help='Classification_weight (default: 1e1)')

    # parser.add_argument('--train_test_split', type=str, default='fixedPatient', metavar='SPL',
    #                     help='aug (default: 0)')
    # parser.add_argument('--paddingMethod', type=str, default='zero', metavar='PAD',
    #                     help='aug (default: 0)')
    args, _ = parser.parse_known_args()
    return args


# debug_path_to_exp = '../../experiment_results/NNI_test'
debug_path_to_exp = './exp-results/NNI_test'
dataset_dir = './data'


tuned_params = nni.get_next_parameter()
if tuned_params == {}:
    debug_mode = True
else:
    debug_mode = False

# exp_type = 'strainmat_to_TOS'
# exp_type = 'multitask_reg_clsDistMap'
# exp_type = 'multitask_reg_cls'
# exp_type = 'scar_cls'
# exp_type = 'scar_distmap_reg'
exp_type = tuned_params.get('exp_type', 'reg')
# exp_type = tuned_params['exp_type']
config = get_default_config(exp_type)
# config['training']['epochs_num'] = 50

# Fetch hyper-parameters from HPO tuner
# comment out following two lines to run the code without NNI framework


modify_config(config, exp_type, tuned_params)
# load_num = -1
load_data_num = None
config['exp_type'] = exp_type
config['data']['train_test_split']['paras']['test_patient_names'] = [
        'Pre_CRT_LBBB_with_scar-49_KJ_MR', 'Pre_CRT_LBBB_with_scar-114_42_BC_MR', 'Pre_CRT_LBBB_with_scar-121_53_DY_MR', 'SET01-CT02', 'SET02-CT28']

if debug_mode:
    
    # path_to_exp = '/home/jrxing/WorkSpace/Research/Cardiac/experiment_results/NNI_test'
    # path_to_exp = '/home/jrxing/Research/Projects/Cardiac/Reperiment_Results/'
    # path_to_exp = '../../experiment_results/NNI_test'
    load_data_num = [20, -20]
    config['data']['train_test_split']['paras']['test_patient_names'] = [
        'SET01-CT02']
    # exp_type = 'multitask-reg-cls'
    exp_type = 'reg'
    config['exp_parent_path'] = debug_path_to_exp
    create_folder(debug_path_to_exp + '/training_results', recursive=False, action_when_exist='pass')
    create_folder(debug_path_to_exp + '/test_results', recursive=False, action_when_exist='pass')
    create_folder(debug_path_to_exp + '/training_results_CAM', recursive=False, action_when_exist='pass')
    create_folder(debug_path_to_exp + '/test_results_CAM', recursive=False, action_when_exist='pass')
    create_folder(debug_path_to_exp + '/networks', recursive=False, action_when_exist='pass')
    create_folder(debug_path_to_exp + '/configs', recursive=False, action_when_exist='pass')
    config['training']['epochs_num'] = 51
    config['training']['batch_size'] = 64
    config['training']['learning_rate'] = 1e-4
    
    # config['data']['input_info'][0]['type'] = 'strainMatFullResolutionSVD'
    config['data']['input_info'] = [{'type': 'strainMatFullResolutionSVD', 'tag': 'strainmat', 'config': {}}]
    # config['data']['input_info'][0]['type'] = 'strainMatFullResolution'
    # config['data']['input_type'] = 'strainMat'
    # config['data']['output_type'] = 'TOS18_Jerry'
    config['data']['output_info'] = [{'type': 'TOS126', 'config':{}, 'tag': 'reg'}]
    # config['data']['output_info'] = [{'type': 'has_scar', 'config':{}}]
    # config['data']['output_info'] = [{'type': 'late_activation_sector_label', 'config':{}}]
    # config['data']['output_info'] = [{'type': 'scar_sector_label', 'config':{}}]
    # config['data']['output_info'] = [{'type': 'late_activation_sector_label', 'config':{}, 'tag': 'cls'}, 
    #                                   {'type':'TOS126', 'config':{}, 'tag': 'reg'}]
    
    
    config['data']['use_data_with_scar'] = 'all'
    # config['data']['scar_free'] = False
    # config['data']['scar_must'] = True
    # config['data']['scar_must_must'] = True
    config['data']['force_onehot'] = False
    config['data']['remove_sector_label_spikes'] = False
    # config['data']['train_test_split']['paras']['test_patient_names'] = ['SET01-CT11', 'SET02-CT28', 'SET03-EC21',
    #                                                                      'SET01-CT02', 'SET01-CT16', 'SET01-CT18']


    # config['data']['filename'] = 'D://dataFull-201-2020-12-23-Jerry.npy'

    # config['data']['filename'] = 'D://dataFull-201-2020-12-23-Jerry.npy'
    # config['data']['filename'] = PurePath('/home/jrxing/WorkSpace/Research/Cardiac/Dataset', 'dataFull-201-2020-12-23-Jerry.npy')
    # config['data']['filename'] = str(PurePath('../../Dataset', 'dataFull-201-2020-12-23-Jerry.npy'))
    # config['data']['input_info'] = update_iotype_argument('strainMatFullResolution')
    # config['data']['TOS_info'] = update_iotype_argument('TOS126')
    
    # config['data']['output_types'] = 'TOSfullRes_Jerry+late_acti_label'.split('+')
    # config['data']['output_types'] = 'TOSfullRes_Jerry+late_acti_label'.split('+')
    # config['data']['output_types'] = 'TOSfullRes_Jerry+strain_curve_type_label'.split('+')
    # config['data']['output_info'] = update_iotype_argument('TOSfullRes_Jerry+scar-AHA-step=50')
    # config['data']['output_info'] = update_iotype_argument('TOS126+scar_sector_percentage')
    # config['data']['output_info'] = update_iotype_argument('TOS126+scar_sector_label')
    # config['data']['output_types'] = 'TOSfullRes_Jerry+strain_curve_type_dist_map'.split('+')
    # config['data']['output_types'] = 'polyfit_coefs+late_acti_label'.split('+')
    # config['data']['output_types'] = 'TOSfullRes_Jerry+late_acti_dist_map'.split('+')
    # config['data']['input_types'] = 'strainMat'.split('+')
    # config['data']['output_types'] = 'TOS18_Jerry+late_acti_label'.split('+')
    # config['data']['output_types'] = 'TOS18_Jerry+late_acti_dist_map'.split('+')
    # config['data']['augmentation'] = update_augment_argument('shift-sector=-32_32_5+mixup=0.2_500', [])
    # aug_args_json_str = '\
    #     {"method":"shift_sector", "paras":"-48,48,3", "include_data_conditions":"has_scar_sector_label"}+\
    #     {"method":"mixup", "paras":"0.5,500", "include_data_conditions":"has_scar_sector_label"}+\
    #     {"method":"shift_sector", "paras":"-32,32,10", "include_data_conditions":"no_scar_sector_label"}+\
    #     {"method":"mixup", "paras":"0.5,100", "include_data_conditions":"no_scar_sector_label"}\
    #     '
    aug_args_json_str = '\
        {"method":"shift_sector", "paras":"-48,48,3", "include_data_conditions":"has_scar"}+\
        {"method":"mixup", "paras":"0.5,500", "include_data_conditions":"has_scar"}+\
        {"method":"shift_sector", "paras":"-32,32,10", "include_data_conditions":"no_scar"}+\
        {"method":"mixup", "paras":"0.5,100", "include_data_conditions":"no_scar"}\
        '
    config['data']['augmentation'] = parse_augment_argument_json(aug_args_json_str)
    # config['data']['augmentation'] = update_augment_argument('shift-sector=-32_32_5+mixup=0.2_500', [])
    # config['data']['augmentation'] = update_augment_argument('shift-sector_-32_32_5', [])
    # print('AUG', update_augment_argument('shift-sector=-32_32_5+mixup=0.1_1000', []))
    # logger = _logger
    # assert 1>10
    # config['data']['train_test_split']['paras']['test_patient_names'] = [
    #     'SET01-CT02']
    
    # config['net'] = {'type': 'NetStrainMat2ClsReg',
    #                  'paras': {
    #                      'n_sector': 18,
    #                      'n_frames': 64,
    #                      'batch_norm': True,
    #                      'activation_func': 'relu',
    #                      'n_conv_layers': 3,
    #                      'n_linear_layers': 3,
    #                      'n_sectors_in': 128,
    #                      'n_sectors_out': 128,
    #                      'n_classes': 1}
    #                  }

    # config['net']['type'] = 'NetStrainMat2ClsReg'
    # config['net']['type'] = 'NetStrainMat2ClsReg'
    config['net']['type'] = 'NetStrainMat2Reg'
    
    
    # CLs network
    # config['net']['paras']['batch_norm'] = True
    # # config['net']['paras']['activation_func'] = 'leaky_relu'
    # config['net']['paras']['activation_func'] = 'relu'
    # config['net']['paras']['conv_layer_num'] = 4
    # config['net']['paras']['pooling_layer_num_max'] = None
    # # config['net']['paras']['n_conv_layers'] = 3
    # config['net']['paras']['linear_layer_num'] = 4
    
    # Multi reg-cls network
    config['net']['paras']['joint_init_conv_channel_num'] = 4
    config['net']['paras']['joint_conv_layer_num'] = 4
    config['net']['paras']['joint_pooling_layer_num_max'] = None
    config['net']['paras']['reg_conv_layer_num'] = 4
    config['net']['paras']['reg_linear_layer_num'] = 2
    config['net']['paras']['reg_pooling_layer_num_max'] = None
    config['net']['paras']['cls_conv_layer_num'] = 4
    config['net']['paras']['cls_linear_layer_num'] = 2
    config['net']['paras']['cls_pooling_layer_num_max'] = None

    
    # config['net']['paras']['force_onehot'] = False
    # config['eval']['paras']['cls_weight'] = 1e2
    # config['eval'] = {'method': 'MSE', 'paras': {}, 'target_tag': 'cls'}
    # config['eval'] = {'method': 'cls', 'paras': {'type': 'Cross Entropy'}, 'target_tag': 'cls'}
    # config['eval'] = {'method': 'cls', 'paras': {'type': 'Negative log likelihood'}, 'target_tag': 'cls'}
    # config['eval'] = {
    #     'method': 'multitask-reg-cls',
    #     'paras': [
    #         {'type': 'MSE', 'weight': 1, 'target_tag': 'reg'},
    #         {'type': 'Negative log likelihood', 'weight': 0, 'target_tag': 'cls'}
    #         ]
    #     }
    config['eval'] = {
        'method': 'reg',
        'paras': [
            {'type': 'MSE', 'weight': 1, 'target_tag': 'reg'}
            ]
        }
    # config['eval'] = [{'method': 'MSE', 'paras': {}, 'target_tag': 'reg'},
    #                   {'method': 'Cross Entropy', 'paras': {}, 'target_tag': 'cls'}]
    # config['data']['aug_more_on_data_with_scar'] = True
    config['show'] = {}
    config['show']['CAM_target_sector'] = 'late_activation_center'
else:
    exp_info_filename = PurePath(str(Path.home()), f'nni-config-{date.today().strftime("%Y-%m")}.yml')
    # exp_info_filename = f'./NNI/configs/nni-config-{date.today().strftime("%Y-%m")}.yml'
    with open(exp_info_filename) as file:
        exp_info = yaml.full_load(file)
    config['exp_parent_path'] = exp_info['exp_parent_path']
# config['data']['filename'] = '../../Dataset/dataFull-201-2020-12-23-Jerry.npy'
# dataset_dir = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry'
# # data_records_filename = str(Path(dataset_dir, 'record_sheets\\cardiac-strainmat-dataset-2021-04-20.xlsx'))
# data_records_filename = str(Path(dataset_dir, 'record_sheets', 'cardiac-strainmat-dataset-2021-06-13-scar-classification.xlsx'))
data_records_filename = tuned_params.get('data_records_filename', 'cardiac-strainmat-dataset-2021-07-04-late-activation-region-classification.xlsx')
data_records_filename_full = str(Path(dataset_dir, 'record_sheets', data_records_filename))
config['data']['filename'] = data_records_filename_full
# config['data']['train_test_split']['paras']['test_patient_names'] = ['SET01-CT11', 'SET02-CT28', 'SET03-EC21',
#                                                                      'SET03-UP34']
# config['data']['train_test_split']['paras']['test_patient_names'] = [
#         'SET01-CT11', 'SET02-CT28', 'SET03-EC21', 'SET03-UP34', 
#         'Pre_CRT_LBBB_with_scar-34_CM_MR', 'Pre_CRT_LBBB_with_scar-86_RS_MR']




config['NNI'] = {
    'experiment_id': nni.get_experiment_id(),
    'trial_id': nni.get_trial_id(),
    'sequence_id': nni.get_sequence_id()
}
config['NNI'][
    'trial_name'] = f"exp-{str(config['NNI']['experiment_id']).strip()}-idx-{config['NNI']['sequence_id']:03d}-trial-{str(config['NNI']['trial_id']).strip()}"

# _logger.info('Hyper-parameters: %s', config)

pprint.pprint(config)

# %% 3. Run Experiment
config = config
NNI = True
logger = _logger
save_model = True
save_prediction = True
save_config = True
trained_model_filename = None

# %% 3.0. Get experiment folder if needed
trail_name = config['NNI'].get('trial_name',
                               f"exp-{str(config['NNI']['experiment_id']).strip()}-idx-{config['NNI']['sequence_id']:03d}-trial-{str(config['NNI']['trial_id']).strip()}")

# %% 3.1. Set device
# gpuIdx = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# %% 4. Load data
if logger is not None:
    logger.info('Load Data')
print('(PRINT) Load Data')

included_data_info = config['data']['input_info'] + config['data']['output_info']
included_data_types = [data_type['type'] for data_type in included_data_info]
dataFilename = config['data']['filename']
dataFull = load_data_from_table(dataFilename, dataset_dir=dataset_dir, data_info=included_data_info, load_num=load_data_num)

from utils.data import get_data_type_by_category
def check_has_scar(datum, strict=False):
    # scar_data_type = get_data_type('scar_sector_percentage')
    if 'scar_sector_percentage' not in datum.keys():
        return False
    
    if strict and np.sum(datum['scar_sector_percentage']) < 0.1:
        return False
    else:
        return True
    
use_data_with_scar = config['data'].get('use_data_with_scar', 'scar_free')
if use_data_with_scar == 'scar_free':
    dataFull = [datum for datum in dataFull if int(datum['hasScar']) == 0]
elif use_data_with_scar == 'scar_must':
    dataFull = [datum for datum in dataFull if datum['hasScar'] != 0]
elif use_data_with_scar == 'scar_must_must':
    dataFull = [datum for datum in dataFull if check_has_scar(datum, strict=True)]
# if config['data'].get('scar_free', False):
#     dataFull = [datum for datum in dataFull if datum['hasScar'] == 0]

# if config['data'].get('scar_must', False):
#     dataFull = [datum for datum in dataFull if datum['hasScar'] != 0]
    
# if config['data'].get('scar_must_must', False):
#     dataFull = [datum for datum in dataFull if np.sum(datum['scar_sector_percentage']) > 0.1]    
# if debug_mode:
#     dataFull = [datum for datum in dataFull if not datum['patient_name'].startswith('Pre_CRT')]
#     config['data']['train_test_split']['paras']['test_patient_names'] = [
#         'SET01-CT11', 'SET02-CT28', 'SET03-EC21', 'SET03-UP34']
from utils.data import report_data_info
report_data_info(dataFull)

# %% 5. Add class label or distance map
from utils.data import add_classification_label
# if config['eval']['method'] in ['multitask-reg-cls', 'cls']:
if exp_type in ['multitask-reg-cls', 'cls']:
    add_classification_label(data=dataFull, data_info=included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False), force_onehot=config['data'].get('force_onehot', True))

# %% 6. Pre-processing
from utils.preprocessing import unify_n_frame, unify_n_sector, remove_last_frames
remove_last_frames(dataFull, included_data_types, n_frames_to_remove=5)
unify_n_frame(dataFull, included_data_types, n_frames_target='power_of_2')
unify_n_sector(dataFull, included_data_types, n_sectors_target='power_of_2', method='copy_boundary')

# %% 7. Add distance map
from utils.data import add_distance_map
if exp_type in ['multitask-reg-clsDistMap', 'reg']:
    add_distance_map(dataFull, included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False))

# %% 7. Train-test split
from utils.data import train_test_split
training_data_raw, test_data = train_test_split(config['data']['train_test_split'], dataFull)
print('training_data_raw len:', len(training_data_raw))
print('test_data len:', len(test_data))

# %% 8. Augmentation
for datum in dataFull:
    datum['augmented'] = False

from utils.augmentation import augment
# if not debug_mode:
training_data_aug, training_data_aug_samples = augment(training_data_raw, included_data_types,
                                                   config['data']['augmentation'])
# else:
    # training_data_aug = []
    # training_data_aug_samples = []
print('training_data_aug len:', len(training_data_aug))
print('training_data_aug_samples len:', len(training_data_aug_samples))

training_data = training_data_raw + training_data_aug

# %% 8.1 Re-label for some types
from utils.data import add_polyfit_coefficient
for data_type in included_data_types:
    if data_type == 'polyfit_coefs':
        add_polyfit_coefficient(training_data + test_data, included_data_types, degree=10)

# %% 9. Set Dataset
from modules.dataset import Dataset
dataset_precision = np.float16
# dataset_precision = np.float32
# dataset_precision = None
training_dataset = Dataset(training_data, config['data']['input_info'], config['data']['output_info'], precision=dataset_precision)
test_dataset = Dataset(test_data, config['data']['input_info'], config['data']['output_info'], precision=dataset_precision)

training_dataset.input_info = config['data']['input_info']
training_dataset.output_info = config['data']['output_info']
test_dataset.input_info = config['data']['input_info']
test_dataset.output_info = config['data']['output_info']

# Reshape data
for data_type in training_dataset.input_types + training_dataset.output_types:
    # print(np.ndim(training_dataset[0][data_type]))
    if data_type in ['TOS126'] and np.ndim(training_dataset[0][data_type]) == 1:
        for datum in training_dataset:
            datum[data_type] = datum[data_type][None, :]
        for datum in test_dataset:
            datum[data_type] = datum[data_type][None, :]

# %% 10. Set Network
from utils.data import get_data_type_by_category, get_data_category_by_type, get_data_info_by_tag
if trained_model_filename is None:
    from modules.networks.get_network import get_network_by_name
    # Input parameters
    # config['net']['paras']['n_sectors_in'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
    # config['net']['paras']['n_sectors_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
    # config['net']['paras']['n_frames'] = training_data[0][config['data']['input_info'][0]['type']].shape[-1]
    config['net']['paras']['input_channel_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-3]
    config['net']['paras']['input_sector_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
    config['net']['paras']['input_frame_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-1]
    # config['net']['paras']['n_sectors_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
    
    
    # Output parameters
    # config['net']['paras']['degree'] = 10
    # if config['eval']['method'] in ['multitask-reg-cls', 'multitask-reg-clsDistMap']:
    if False:
        if exp_type in ['multitask-reg-cls', 'multitask-reg-clsDistMap']:
            if exp_type in ['multitask-reg-cls']:
                cls_out_data_type = get_data_type_by_category('sector_label', config['data']['output_info'])
                # cls_out_data_type = get_data_info_by_tag('cls', config['data']['output_info'])['type']
                config['net']['paras']['n_classes'] = training_data[0][cls_out_data_type].shape[-2]
            elif config['eval']['method'] in ['multitask-reg-clsDistMap']:
                cls_out_data_type = get_data_type_by_category('sector_dist_map', config['data']['output_info'])
                config['net']['paras']['n_classes'] = training_data[0][cls_out_data_type].shape[-2]
            # config['net']['paras']['n_sectors_out'] = 22
            config['net']['paras']['reg_n_dim_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
            config['net']['paras']['cls_n_dim_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2] * \
                                                      training_data[0][cls_out_data_type].shape[-2]    
            config['net']['paras']['force_onehot'] = config['data'].get('force_onehot', True)
        # elif config['eval']['method'] in ['cls']:
        elif exp_type in ['cls']:
            cls_out_data_type = get_data_type_by_category('sector_label', config['data']['output_info'])
            if cls_out_data_type is not None:
                config['net']['paras']['n_classes'] = training_data[0][cls_out_data_type].shape[-2]
                config['net']['paras']['cls_n_dim_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2] * \
                                                        training_data[0][cls_out_data_type].shape[-2]
            else:
                cls_out_data_type = get_data_type_by_category('data_label', config['data']['output_info'])
                config['net']['paras']['n_classes'] = training_data[0][cls_out_data_type].shape[-2]
                config['net']['paras']['cls_n_dim_out'] = training_data[0][cls_out_data_type].shape[-2]
            config['net']['paras']['force_onehot'] = config['data'].get('force_onehot', True)
        # elif config['eval']['method'] in ['reg']:
        elif exp_type in ['reg']:
            try:
                reg_out_data_type = get_data_type_by_category('sector_dist_map', config['data']['output_info'])
            except:
                reg_out_data_type = get_data_type_by_category('sector_value', config['data']['output_info'])
                if get_data_category_by_type(reg_out_data_type) == 'TOS':
                    config['net']['paras']['add_last_relu'] = True
            config['net']['paras']['n_classes'] = training_data[0][reg_out_data_type].shape[-2]
            config['net']['paras']['reg_n_dim_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2] * \
                                                      training_data[0][reg_out_data_type].shape[-2]
        else:
            raise ValueError(f'Unsupported exp type: {exp_type}')
                                                  
    if config['net']['type'] in ['NetStrainMat2Cls']:
        cls_out_data_type = get_data_type_by_category('sector_label', config['data']['output_info'])
        classes_num = training_data[0][cls_out_data_type].shape[-2]
        config['net']['paras']['input_sector_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
        config['net']['paras']['input_frame_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-1]
        config['net']['paras']['force_onehot'] = config['data'].get('force_onehot', True)
        # if not config['data'].get('force_onehot', True) and classes_num == 2:
        if not config['data'].get('force_onehot', True) and classes_num <= 2:
            config['net']['paras']['cls_output_dim'] = config['net']['paras']['input_sector_num']
        else:
            config['net']['paras']['cls_output_dim'] = \
                config['net']['paras']['input_sector_num'] * classes_num
                
    elif config['net']['type'] in ['NetStrainMat2ClsReg']:
        cls_out_data_type = get_data_info_by_tag('cls', config['data']['output_info'])['type']
        classes_num = training_data[0][cls_out_data_type].shape[-2]
        # config['net']['paras']['input_sector_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
        # config['net']['paras']['input_frame_num'] = training_data[0][config['data']['input_info'][0]['type']].shape[-1]
        config['net']['paras']['cls_force_onehot'] = config['data'].get('force_onehot', True)
        config['net']['paras']['cls_class_normlize_layer'] = config['net']['paras'].get('cls_class_normlize_layer', 'log softmax')
        
        # if not config['data'].get('force_onehot', True) and classes_num == 2:
        if not config['data'].get('force_onehot', True) and classes_num <= 2:
            config['net']['paras']['cls_output_dim'] = config['net']['paras']['input_sector_num']
        else:
            config['net']['paras']['cls_output_dim'] = \
                config['net']['paras']['input_sector_num'] * classes_num
        config['net']['paras']['reg_output_dim'] = config['net']['paras']['input_sector_num']
        reg_out_data_type = get_data_info_by_tag('reg', config['data']['output_info'])['type']
        if np.ndim(training_data[0][reg_out_data_type]) == 2:
            config['net']['paras']['reg_output_additional_dim'] = False
        else:
            config['net']['paras']['reg_output_additional_dim'] = True
    elif config['net']['type'] in ['NetStrainMat2Reg']:
        reg_out_data_type = get_data_info_by_tag('reg', config['data']['output_info'])['type']
        strainmat_data_type = get_data_info_by_tag('strainmat', config['data']['input_info'])['type']
        classes_num = training_data[0][reg_out_data_type].shape[-2] if np.ndim(training_data[0][reg_out_data_type]) >= 3 else 1
        
        config['net']['paras']['input_sector_num'] = training_data[0][strainmat_data_type].shape[-2]
        config['net']['paras']['input_frame_num'] = training_data[0][strainmat_data_type].shape[-1]
        config['net']['paras']['force_onehot'] = config['data'].get('force_onehot', False)
        config['net']['paras']['classes_num'] = classes_num        
        
        # if not config['data'].get('force_onehot', True) and classes_num == 2:
        if not config['data'].get('force_onehot', False) and classes_num <= 2:
            config['net']['paras']['reg_output_dim'] = config['net']['paras']['input_sector_num']
        else:
            config['net']['paras']['reg_output_dim'] = \
                config['net']['paras']['input_sector_num'] * classes_num
            
        
    network = get_network_by_name(config['net']['type'], config['net'])
    network.set_input_types(training_dataset.input_types, [datum_info['tag'] for datum_info in config['data']['input_info']])
    network.set_output_types(training_dataset.output_types, [datum_info['tag'] for datum_info in config['data']['output_info']])
    network.to(device)
    print(network)
else:
    network = torch.load(trained_model_filename)


# %% 11. Set Network Module
from modules.net import NetModule
net = NetModule(network=network, evaluation_config=config['eval'], regularization_config=config['regularization'],
                device=device)

# %% 12. Training
if trained_model_filename is None:
    net.network.train()
    training_loss_final, training_loss_history, valid_loss_final, valid_loss_history, valid_reg_loss_final, past_time = \
        net.train(training_dataset=training_dataset, valid_dataset=test_dataset, training_config=config['training'],
                  NNI=NNI, logger=logger)
else:
    training_loss_final, valid_loss_final = None, None

# %% 13. Save Prediction Results
# Prediction
net.network.eval()
# training_data_to_save = training_data_raw[:12] + training_data_aug[:5] + training_data_aug_samples
training_patients_to_show = ['SET01-CT02', 'SET01-CT16']
training_data_to_save = training_data_raw[::5][:12] + [d for d in training_data_raw if d['patient_name'] in training_patients_to_show]
# training_dataset_raw = [datum for datum in training_dataset if datum['augmented'] == False]
# training_data_to_save = training_dataset_raw[::5][:12] + [d for d in training_dataset_raw if d['patient_name'] in training_patients_to_show]
for datum in training_data_raw + test_data + training_data_to_save:
    datum_pred = net.pred(datum, training_dataset.input_types, training_dataset.output_info)
    for output_data_key in datum_pred['data'].keys():
        datum[output_data_key + '_pred'] = datum_pred['data'][output_data_key]
    datum['loss_pred'] = datum_pred['loss']

# %%
show_config = config.get('show', {})
# expPath = '../nni_logs/' + config['NNI']['experiment_id'] + '/trials/' + config['NNI']['trial_id'] + '/'
from utils.plot import save_multi_strainmat_with_curves
from utils.plot import save_multi_strainmat_with_curves_and_activation_map
curve_types_to_plot = []
for output_data_type in [data_info['type'] for data_info in config['data']['output_info']]:
    curve_types_to_plot.append(output_data_type)
    curve_types_to_plot.append(output_data_type + '_pred')


if save_prediction:
    save_test_results_filename_full = str(
            PurePath(config['exp_parent_path'], 'test_results', f"{trail_name}-test-results.pdf"))
    save_training_results_filename_full = str(
            PurePath(config['exp_parent_path'], 'training_results', f"{trail_name}-train-results.pdf"))
    
    strainmat_type = config['data']['input_info'][0]['type']
    save_multi_strainmat_with_curves(data=test_data,
                                     strainmat_type=strainmat_type,
                                     curve_types=curve_types_to_plot,
                                     legends=curve_types_to_plot,
                                     save_filename=save_test_results_filename_full,
                                     subtitles=[datum['patient_slice_name'] for datum in test_data])

    
    save_multi_strainmat_with_curves(data=training_data_to_save,
                                     strainmat_type=strainmat_type,
                                     curve_types=curve_types_to_plot,
                                     legends=curve_types_to_plot,
                                     save_filename=save_training_results_filename_full,
                                     subtitles=[datum['patient_slice_name'] for datum in training_data_to_save])


debug_show_accuracy = False
if debug_show_accuracy:
    # training_label = [int(datum['has_scar'][0,-1,0]) for datum in training_data_to_save]
    # training_label_pred = [int(datum['has_scar_pred'][0,-1,0]>=0.5) for datum in training_data_to_save]
    # test_label = [int(datum['has_scar'][0,-1,0]) for datum in test_data]
    # test_label_pred = [int(datum['has_scar_pred'][0,-1,0]>=0.5) for datum in test_data]
    training_label = [int(datum['has_scar'][0,0]) for datum in training_data_to_save]
    training_label_pred = [int(datum['has_scar_pred'][0,0]>=0.5) for datum in training_data_to_save]
    test_label = [int(datum['has_scar'][0,0]) for datum in test_data]
    test_label_pred = [int(datum['has_scar_pred'][0,0]>=0.5) for datum in test_data]

debug_plot_pred = False
if debug_plot_pred:    
    from utils.plot import plot_multi_strainmat_with_curves
    plot_multi_strainmat_with_curves(
        data=training_data_to_save, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
        fig=None, axs=None, 
        legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in training_data_to_save], 
        vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        colors=None)
    plot_multi_strainmat_with_curves(
        data=test_data, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
        fig=None, axs=None, 
        legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in test_data], 
        vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        colors=None)

debug_plot_CAM = False
if debug_plot_CAM:
    from utils.CAM_utils import get_target_sector
    from modules.visualization.pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
    # from utils.scar_utils import find_connected_components_binary_1d
    net.network.double()
    linear_layer_idx = np.where([type(layer) is torch.nn.modules.linear.Linear for layer in net.network.layers])[0]
    if type(net.network.layers[linear_layer_idx[0] - 1]) is torch.nn.modules.flatten.Flatten:
        layer_before_first_linear = net.network.layers[linear_layer_idx[0] - 2]
    else:
        layer_before_first_linear = net.network.layers[linear_layer_idx[0] - 1]
    conv_layers = [layer for layer in net.network.layers if type(layer) is torch.nn.modules.conv.Conv2d]
    linear_layers = [layer for layer in net.network.layers if type(layer) is torch.nn.modules.linear.Linear]
    last_conv_layer = conv_layers[-1]
    first_conv_layer = conv_layers[0]
    target_layer = last_conv_layer
    # target_layer = layer_before_first_linear
    cam = GradCAM(model=net.network, target_layer=target_layer, use_cuda=True)
    
    if exp_type in ['reg']:
        cam_outout_type = 'reg'
    elif exp_type in ['cls']:
        cam_outout_type = 'cls'
        
    
    for target_category in [-1]:
        ic(target_category)
        # for datum in training_data_raw + test_data + training_data_to_save:
        for datum in test_data + training_data_to_save:
            input_data = datum
            input_tensor = torch.tensor(datum[config['data']['input_info'][0]['type']])
            # scar_regions = find_connected_components_binary_1d((datum['scar_sector_label']).astype(np.int)[0,1,:], order = 'size')[0]
            # if len(scar_regions) > 0:
            #     largest_scar_region_center = scar_regions[0]['center']
            # else:
            #     largest_scar_region_center = 60
            target_sector_type = show_config.get('CAM_target_sector', 'center_sector')
            target_sector = get_target_sector(datum, config['data']['output_info'][0]['type'], target_sector_type)
            # target_sector = 0
            
            # datum_pred = net.pred(datum, training_dataset.input_types, training_dataset.output_types)
            cam_scar_center = cam(input_datum = input_data, 
                                  input_types = training_dataset.input_types,
                                  output_types = training_dataset.output_types,
                                  device = device,
                                  task_type = cam_outout_type, 
                                  sector_idx = target_sector, 
                                  target_category=target_category, 
                                  aug_smooth=False, eigen_smooth=False, 
                                  counter_factual=False,
                                  evaluation_config = config['eval'])
            
            # cam_scar_center = cam(input_tensor=input_tensor, output_type = cam_outout_type, sector_idx = largest_scar_region_center, target_category=target_category, aug_smooth=False, eigen_smooth=True, counter_factual=False)
            # counter_cam_scar_center = cam(input_tensor=input_tensor, output_type = cam_outout_type, sector_idx = largest_scar_region_center, target_category=target_category, aug_smooth=False, eigen_smooth=True, counter_factual=True)
            datum['cam'] = cam_scar_center
            datum['cam']['sector'] = target_sector
            
        from utils.plot import plot_multi_strainmat_with_curves_and_activation_map
        data_to_show = test_data[:4]
        # data_to_show = test_data[-4:]
        # data_to_show = training_data_to_save[:4]
        plot_multi_strainmat_with_curves_and_activation_map(
            data=data_to_show, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
            cam_data_types=training_dataset.output_types + ['total_loss'], counter_cam_data_types=training_dataset.output_types, 
            fig=None, axs=None, 
            legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in data_to_show], 
            vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
            colors=None,
            check_activation_sectors = [datum['cam']['sector'] for datum in data_to_show])
        
    

save_CAM = False
if save_CAM:        
    save_test_results_CAM_filename_full = str(
            PurePath(config['exp_parent_path'], 'test_results_CAM', f"{trail_name}-test-results.png"))
    save_training_results_CAM_filename_full = str(
            PurePath(config['exp_parent_path'], 'training_results_CAM', f"{trail_name}-train-results.png"))
    
    from modules.visualization.pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
    from utils.scar_utils import find_connected_components_binary_1d
    net.network.double()
    
    last_conv_layer = [layer for layer in net.network.layers if type(layer) is torch.nn.modules.conv.Conv2d][-1]
    target_layer = last_conv_layer
    cam = GradCAM(model=net.network, target_layer=target_layer, use_cuda=True)
    
    if exp_type in ['reg']:
        cam_outout_type = 'reg'
    elif exp_type in ['cls']:
        cam_outout_type = 'cls'
    
    target_category = -1
    # for datum in training_data_raw + test_data + training_data_to_save:
    for datum in test_data + training_data_to_save:
        input_data = datum
        input_tensor = torch.tensor(datum[config['data']['input_info'][0]['type']])
        
        # if any([scar_type in training_dataset.output_types for scar_type in ['scar_sector_label', 'scar_sector_distmap']]):
        #     target_sector = get_target_sector(datum, config['data']['output_info'][0]['type'], 'scar_center')
        # else:
        #     target_sector = get_target_sector(datum, config['data']['output_info'][0]['type'], 'difference')
        target_sector_type = show_config.get('CAM_target_sector', 'center_sector')
        target_sector = get_target_sector(datum, config['data']['output_info'][0]['type'], target_sector_type)
        
        # datum_pred = net.pred(datum, training_dataset.input_types, training_dataset.output_types)
        cam_scar_center = cam(input_datum = input_data, 
                              input_types = training_dataset.input_types,
                              output_types = training_dataset.output_types,
                              device = device,
                              task_type = cam_outout_type, 
                              sector_idx = target_sector, 
                              target_category=target_category, 
                              aug_smooth=False, eigen_smooth=True, 
                              counter_factual=False,
                              evaluation_config = config['eval'])
        
        # cam_scar_center = cam(input_tensor=input_tensor, output_type = cam_outout_type, sector_idx = largest_scar_region_center, target_category=target_category, aug_smooth=False, eigen_smooth=True, counter_factual=False)
        # counter_cam_scar_center = cam(input_tensor=input_tensor, output_type = cam_outout_type, sector_idx = largest_scar_region_center, target_category=target_category, aug_smooth=False, eigen_smooth=True, counter_factual=True)
        datum['cam'] = cam_scar_center
        datum['cam']['sector'] = target_sector
    
    save_multi_strainmat_with_curves_and_activation_map(
        data=test_data, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
        save_filename = save_test_results_CAM_filename_full,
        cam_data_types=training_dataset.output_types + ['total_loss'], counter_cam_data_types=training_dataset.output_types, 
        fig=None, axs=None, 
        legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in test_data], 
        vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        colors=None,
        check_activation_sectors = [datum['cam']['sector'] for datum in test_data])
    
    save_multi_strainmat_with_curves_and_activation_map(
        data=training_data_to_save, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
        save_filename = save_training_results_CAM_filename_full,
        cam_data_types=training_dataset.output_types + ['total_loss'], counter_cam_data_types=training_dataset.output_types, 
        fig=None, axs=None, 
        legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in training_data_to_save], 
        vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        colors=None,
        check_activation_sectors = [datum['cam']['sector'] for datum in training_data_to_save])
        
# %% 14. Save Config
config['performance'] = {}
config['performance']['training_loss'] = training_loss_final
config['performance']['valid_loss'] = valid_loss_final
# config['performance']['dafault_loss']  = valid_loss_final
config['performance']['dafault_loss'] = valid_reg_loss_final
if save_config:
    save_config_filename_full = str(PurePath(config['exp_parent_path'], 'configs', f"{trail_name}-config.json"))

    # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(save_config_filename_full, 'w') as file:
        # yaml.dump(config, file)
        json.dump(config, file, cls=NumpyEncoder)

# return network, config['performance'], training_data_raw, test_data

# %% Load performance log file
# exp_trial_performance_log_filename = PurePath(config['exp_parent_path'], 'trial_performance_log.yml')
exp_trial_performance_log_filename = config['exp_parent_path'] + 'trial_performance_log.yml'
try:
    with open(exp_trial_performance_log_filename, 'r') as yamlfile:
        exp_trial_performance_log = yaml.safe_load(yamlfile)  # Note the safe_load
except:
    exp_trial_performance_log = None

if exp_trial_performance_log is None:
    exp_trial_performance_log = {config['NNI']['trial_id']: config['performance']}
    trail_is_first = True
else:
    exp_trial_performance_log[config['NNI']['trial_id']] = config['performance']
    trail_is_first = False

# Save network model if current performance is better than the best one
if trail_is_first:
    save_model = True
else:
    valid_performance_log = [exp_trial_performance_log[trial_id]['dafault_loss'] for trial_id in
                             exp_trial_performance_log.keys()]
    if config['performance']['dafault_loss'] <= min(valid_performance_log):
        save_model = True
    else:
        save_model = False

if save_model:
    save_model_filename = f"{config['NNI']['trial_name']}-network.pth"
    save_model_filename_full = str(PurePath(config['exp_parent_path'], 'networks', save_model_filename))
    # torch.save(network, save_model_filename_full)
    net.save_network(save_model_filename_full)

# Save current performance
with open(exp_trial_performance_log_filename, 'w') as yamlfile:
    yaml.safe_dump(exp_trial_performance_log, yamlfile)  # Also note the safe_dump
    
#%%
