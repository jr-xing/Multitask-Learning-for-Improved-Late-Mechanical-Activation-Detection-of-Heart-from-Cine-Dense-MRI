# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:51:31 2021

@author: Jerry Xing
"""

# %%1. Load Config
# import json
import yaml
from yaml import Loader
from pathlib import Path
network_store_dir = "./trained_networks"
network_name = 'exp-KObU6YMd-idx-010-trial-AuuSz'
config_filename = str(Path(network_store_dir, network_name + '-config.json'))
config = yaml.dump(yaml.load(config_filename, Loader=Loader))

with open(config_filename, 'r') as stream:
    # try:
    config = yaml.safe_load(stream)
    # except yaml.YAMLError as exc:
    #     print(exc)

#%% 2. Load Data
load_training_data = False
load_test_data = True


from utils.io import load_data_from_table
dataset_dir = './data'
# load_data_num = [20, -20]
load_data_num = None
included_data_info = config['data']['input_info'] + config['data']['output_info']
included_data_types = [data_type['type'] for data_type in included_data_info]
dataFilename = config['data']['filename']

if load_training_data == False and load_test_data == True:
    load_patient_names = config['data']['train_test_split']['paras']['test_patient_names']
else:
    load_patient_names = None
dataFull = load_data_from_table(dataFilename, 
                                dataset_dir=dataset_dir, 
                                data_info=included_data_info, 
                                load_num=load_data_num,
                                include_patient_names = load_patient_names)
    
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

# Generate data if needed
# Add label
exp_type = config.get('exp_type', 'reg')
from utils.data import add_classification_label
if exp_type in ['multitask-reg-cls', 'cls']:
    add_classification_label(data=dataFull, data_info=included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False), force_onehot=config['data'].get('force_onehot', True))

# Pre-processing
from utils.preprocessing import unify_n_frame, unify_n_sector, remove_last_frames
remove_last_frames(dataFull, included_data_types, n_frames_to_remove=5)
unify_n_frame(dataFull, included_data_types, n_frames_target='power_of_2')
unify_n_sector(dataFull, included_data_types, n_sectors_target='power_of_2', method='copy_boundary')

# Add distance map
from utils.data import add_distance_map
if exp_type in ['multitask-reg-clsDistMap', 'reg']:
    add_distance_map(dataFull, included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False))
    

if load_training_data == True:
    # Train-test split
    from utils.data import train_test_split
    training_data_raw, test_data = train_test_split(config['data']['train_test_split'], dataFull)
    print('training_data_raw len:', len(training_data_raw))
    print('test_data len:', len(test_data))
    
    # Augmentation

    for datum in dataFull:
        datum['augmented'] = False
    
    from utils.augmentation import augment
    # if not debug_mode:
    training_data_aug, training_data_aug_samples = augment(training_data_raw, included_data_types,
                                                       config['data']['augmentation'])
    print('training_data_aug len:', len(training_data_aug))
    print('training_data_aug_samples len:', len(training_data_aug_samples))
    training_data = training_data_raw + training_data_aug

    # Set Dataset
    from modules.dataset import Dataset
    import numpy as np
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
    
    training_data = training_data_raw + training_data_aug
else:
    from modules.dataset import Dataset
    import numpy as np
    dataset_precision = np.float16
    
    trainig_data = []
    test_data = dataFull
    test_dataset = Dataset(test_data, config['data']['input_info'], config['data']['output_info'], precision=dataset_precision)
    test_dataset.input_info = config['data']['input_info']
    test_dataset.output_info = config['data']['output_info']

# %% 2. Load network
import os
import torch
from modules.net import NetModule
from modules.networks.get_network import get_network_by_name
network_filename = str(Path(network_store_dir, network_name + '-network.pth'))
# network_filename = str(Path(network_store_dir, network_name + '-network-retrain.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


network = get_network_by_name(config['net']['type'], config=config['net'])
network.set_input_types([datum_info['type'] for datum_info in config['data']['input_info']], [datum_info['tag'] for datum_info in config['data']['input_info']])
network.set_output_types([datum_info['type'] for datum_info in config['data']['output_info']], [datum_info['tag'] for datum_info in config['data']['output_info']])
network.to(device)

net = NetModule(network=network, evaluation_config=config['eval'], regularization_config=config['regularization'],
                device=device)
if os.path.isfile(network_filename):
    net.load_network(network_filename)
else:
    # Load data
    from utils.io import load_data_from_table
    dataset_dir = '../../Dataset/CRT_TOS_Data_Jerry'
    # load_data_num = [20, -20]
    load_data_num = None
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
    
    # Generate data if needed
    # Add label
    exp_type = config.get('exp_type', 'reg')
    from utils.data import add_classification_label
    if exp_type in ['multitask-reg-cls', 'cls']:
        add_classification_label(data=dataFull, data_info=included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False), force_onehot=config['data'].get('force_onehot', True))
    
    # Pre-processing
    from utils.preprocessing import unify_n_frame, unify_n_sector, remove_last_frames
    remove_last_frames(dataFull, included_data_types, n_frames_to_remove=5)
    unify_n_frame(dataFull, included_data_types, n_frames_target='power_of_2')
    unify_n_sector(dataFull, included_data_types, n_sectors_target='power_of_2', method='copy_boundary')
    
    # Add distance map
    from utils.data import add_distance_map
    if exp_type in ['multitask-reg-clsDistMap', 'reg']:
        add_distance_map(dataFull, included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False))
        
    # Train-test split
    from utils.data import train_test_split
    training_data_raw, test_data = train_test_split(config['data']['train_test_split'], dataFull)
    print('training_data_raw len:', len(training_data_raw))
    print('test_data len:', len(test_data))
    
    # Augmentation
    for datum in dataFull:
        datum['augmented'] = False
    
    from utils.augmentation import augment
    # if not debug_mode:
    training_data_aug, training_data_aug_samples = augment(training_data_raw, included_data_types,
                                                       config['data']['augmentation'])
    print('training_data_aug len:', len(training_data_aug))
    print('training_data_aug_samples len:', len(training_data_aug_samples))
    training_data = training_data_raw + training_data_aug
    
    # Set Dataset
    from modules.dataset import Dataset
    import numpy as np
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
    
    training_data = training_data_raw + training_data_aug

    
    net.network.train()
    training_loss_final, training_loss_history, valid_loss_final, valid_loss_history, valid_reg_loss_final, past_time = \
        net.train(training_dataset=training_dataset, valid_dataset=test_dataset, training_config=config['training'],
                  NNI=False, logger=None)

#%% Savenetwork
from pathlib import PurePath
save_model_filename = f"{config['NNI']['trial_name']}-network-retrain.pth"
# save_model_filename_full = str(PurePath(config['exp_parent_path'], 'networks', save_model_filename))
save_model_filename_full = str(PurePath(network_store_dir, save_model_filename))
# torch.save(network, save_model_filename_full)
net.save_network(save_model_filename_full)

# import torch
# torch.save({
#     'model_state_dict': network.state_dict(),
#     'optimizer_state_dict': net.optimizer.state_dict(),
#     }, save_model_filename_full)

# torch.save(network, str(PurePath(network_store_dir, f"{config['NNI']['trial_name']}-network-retrain-FULL.pth")))
# %% 3. Prepare input data
# 1) Load raw data
# from utils.io import load_data_from_table
# check_test_data_only = True
# if check_test_data_only:
#     include_patient_names = config['data']['train_test_split']['paras']['test_patient_names']
# else:
#     include_patient_names = None

# included_data_info = config['data']['input_info'] + config['data']['output_info']
# included_data_types = [data_type['type'] for data_type in included_data_info]
# exp_type = config.get('confog_type', 'multitask-reg-cls')

# dataFull = load_data_from_table(data_records_filename = config['data']['filename'], 
#                             data_info = config['data']['input_info'] + config['data']['output_info'],
#                             include_patient_names = include_patient_names)

 
# # 2) Prepare additional data if necessary
# from utils.data import add_classification_label
# # if config['eval']['method'] in ['multitask-reg-cls', 'cls']:
# if exp_type in ['multitask-reg-cls', 'cls']:
#     add_classification_label(data = dataFull, 
#                              data_info = included_data_info, 
#                              remove_spikes = config['data'].get('remove_sector_label_spikes', False), 
#                              force_onehot = config['data'].get('force_onehot', True))
    
    
# # 6. Pre-processing
# from utils.preprocessing import unify_n_frame, unify_n_sector, remove_last_frames
# remove_last_frames(dataFull, included_data_types, n_frames_to_remove=5)
# unify_n_frame(dataFull, included_data_types, n_frames_target='power_of_2')
# unify_n_sector(dataFull, included_data_types, n_sectors_target='power_of_2', method='copy_boundary')

# # 7. Add distance map
# from utils.data import add_distance_map
# if exp_type in ['multitask-reg-clsDistMap', 'reg']:
#     add_distance_map(dataFull, included_data_info, remove_spikes=config['data'].get('remove_sector_label_spikes', False))
    
# %% Prediction
net.network.eval()
training_data_to_save = []
# test_data = test_data
for datum in test_data + training_data_to_save:
    datum_pred = net.pred(datum, [data_type['type'] for data_type in config['data']['input_info']], config['data']['output_info'])
    for output_data_key in datum_pred['data'].keys():
        datum[output_data_key + '_pred'] = datum_pred['data'][output_data_key]
    datum['loss_pred'] = datum_pred['loss']
    
#%% Save prediction
# config['NNI']['trial_name']
import numpy as np
save_test_data_filename = f"{config['NNI']['trial_name']}-test-data.npy"
save_test_data_filename_full = str(PurePath(network_store_dir, save_test_data_filename))
np.save(save_test_data_filename_full, test_data)

# %% Combine reg and cls result
output_data_tags = [term['tag'] for term in config['data']['output_info']]
if 'cls' in output_data_tags and 'reg' in output_data_tags:
    from utils.data import get_data_info_by_tag
    import numpy as np
    cls_data_type = get_data_info_by_tag('cls', config['data']['output_info'])['type']
    reg_data_type = get_data_info_by_tag('reg', config['data']['output_info'])['type']
    combined_type = 'combined'
    for datum in test_data:
        cls_arr = np.argmax(datum[cls_data_type], axis=1) != 0
        cls_pred_arr = np.argmax(datum[cls_data_type], axis=1) != 0
        datum[combined_type] = datum[reg_data_type].copy()
        datum[combined_type][0, np.logical_not(cls_arr[0,:])] = 17
        
        datum[combined_type + '_pred'] = datum[reg_data_type + '_pred'].copy()
        datum[combined_type + '_pred'][0,np.logical_not(cls_pred_arr[0,:])] = 17
        # datum[combined_type] = datum[reg_data_type] * (datum[cls_data_type][:, -1, :] >= 0.5)
        # datum[combined_type + '_pred'] = datum[reg_data_type + '_pred'] * (datum[cls_data_type + '_pred'][:, -1, :] >= 0.5)

from utils.plot import plot_multi_strainmat_with_curves
show_combined = True
if show_combined:
    curve_types_to_show = [combined_type, combined_type + '_pred']
else:
    curve_types_to_show = [reg_data_type, reg_data_type + '_pred']

if len(training_data_to_save) > 0:
    plot_multi_strainmat_with_curves(
        data=training_data_to_save, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_show, 
        fig=None, axs=None, 
        legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in training_data_to_save], 
        vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        colors=None)
if len(test_data) > 0:
    plot_multi_strainmat_with_curves(
        data=test_data, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_show, 
        fig=None, axs=None, 
        legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in test_data], 
        vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        colors=None)
# %% Create distance map for regression gt and output
output_data_tags = [term['tag'] for term in config['data']['output_info']]
if 'cls' in output_data_tags:
    from utils.data import get_data_info_by_tag, generate_distance_map
    import numpy as np    
    cls_data_type = get_data_info_by_tag('cls', config['data']['output_info'])['type']
    distmap_type = 'distmap'
    for datum in dataFull:
        cls_arr = (np.argmax(datum[cls_data_type], axis=1) != 0).flatten().astype(int)
        cls_pred_arr = (np.argmax(datum[cls_data_type], axis=1) != 0).flatten().astype(int)
        
        distmap = generate_distance_map(cls_arr)
        distmap_pred = generate_distance_map(cls_pred_arr)
        
        datum[distmap_type] = distmap        
        datum[distmap_type + '_pred'] = distmap_pred

from utils.TOS3DPlotInterpFunc import generate_3D_Activation_map
target_patient_name = 'Pre_CRT_LBBB_with_scar-121_53_DY_MR'
data_to_show_3D_activation_map = [datum for datum in dataFull if datum['patient_name'] == target_patient_name]
generate_3D_Activation_map(data = data_to_show_3D_activation_map,
                    tos_key = distmap_type,
                    spatial_location_key = 'slice_spatial_location', 
                    slice_spatial_order = 'decreasing',
                    title = 'Ground Truth',
                    vmax = None, vmin = None)
generate_3D_Activation_map(data = data_to_show_3D_activation_map,
                    tos_key = distmap_type + '_pred',
                    spatial_location_key = 'slice_spatial_location', 
                    slice_spatial_order = 'decreasing',
                    title = 'Prediction',
                    vmax = None, vmin = None)
# %% Show prediction
from utils.plot import save_multi_strainmat_with_curves
from utils.plot import save_multi_strainmat_with_curves_and_activation_map

curve_types_to_plot = []
for output_data_type in [data_info['type'] for data_info in config['data']['output_info']]:
    curve_types_to_plot.append(output_data_type)
    curve_types_to_plot.append(output_data_type + '_pred')

debug_plot_pred = True
if debug_plot_pred:    
    from utils.plot import plot_multi_strainmat_with_curves
    if len(training_data_to_save) > 0:
        plot_multi_strainmat_with_curves(
            data=training_data_to_save, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
            fig=None, axs=None, 
            legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in training_data_to_save], 
            vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
            colors=None)
    if len(test_data) > 0:
        plot_multi_strainmat_with_curves(
            data=test_data, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
            fig=None, axs=None, 
            legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in test_data], 
            vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
            colors=None)
# %% 3D Activation map
debug_show_3D_activation_map = True
if debug_show_3D_activation_map:
    from utils.io import loadmat
    from utils.DENSE_utils import spl2patchSA    
    # for datum in dataFull:
    for datum in test_data:
        datum['AnalysisFv'] = spl2patchSA(loadmat(datum['DENSE_filename']))
    
    # target_patient_name = 'SET02-CT28'
    target_patient_name = 'SET01-CT11'
    data_to_show_3D_activation_map = [datum for datum in dataFull if datum['patient_name'] == target_patient_name]
    # data_to_show_3D_activation_map = [data_to_show_3D_activation_map[idx] for idx in [0,1,3]]
    # data_to_show_3D_activation_map = [data_to_show_3D_activation_map[idx] for idx in [0,1,3,5,6,7]]
    data_to_show_3D_activation_map = [data_to_show_3D_activation_map[idx] for idx in range(9) if idx not in [7]]
    # from utils.TOS3DPlotInterpFunc import TOS3DPlotInterp_OLD
    # vmax = None
    # vmax = 50
    c_type = 'combined'
    # c_type = 'TOS126'
    # TOS3DPlotInterp_OLD(dataOfPatient = data_to_show_3D_activation_map,
    #                     tos_key = c_type,
    #                     spatial_location_key = 'slice_spatial_location', 
    #                     title = 'GT',
    #                     vmax = 50)
    # TOS3DPlotInterp_OLD(dataOfPatient = data_to_show_3D_activation_map,
    #                     tos_key = c_type + '_pred',
    #                     spatial_location_key = 'slice_spatial_location', 
    #                     title = 'Prediction',
    #                     vmax = None)
    
    # from utils.TOS3DPlotInterpFunc import generate_3D_Activation_map
    # generate_3D_Activation_map(data = data_to_show_3D_activation_map,
    #                     tos_key = c_type,
    #                     spatial_location_key = 'slice_spatial_location', 
    #                     slice_spatial_order = 'decreasing',
    #                     title = 'Ground Truth',
    #                     vmax = None, vmin = None)
    # generate_3D_Activation_map(data = data_to_show_3D_activation_map,
    #                     tos_key = c_type + '_pred',
    #                     spatial_location_key = 'slice_spatial_location', 
    #                     slice_spatial_order = 'decreasing',
    #                     title = 'Prediction',
    #                     vmax = None, vmin = None)
    
    from utils.TOS3DPlotInterpFunc import generate_3D_Activation_map
    # vmax = None
    # vmin = None
    # interpolate_3D_activation_map = False
    interpolate_3D_activation_map = True
    generate_3D_Activation_map(data = data_to_show_3D_activation_map,
                        tos_key = 'TOS126',
                        spatial_location_key = 'slice_spatial_location', 
                        slice_spatial_order = 'increasing',
                        title = 'Ground Truth',
                        vmax = 60, vmin = 17,
                        interpolate = interpolate_3D_activation_map,
                        align_centers=True)
    generate_3D_Activation_map(data = data_to_show_3D_activation_map,
                        tos_key = 'TOS126' + '_pred',
                        spatial_location_key = 'slice_spatial_location', 
                        slice_spatial_order = 'increasing',
                        title = 'Prediction',
                        vmax = 60, vmin = 17,
                        interpolate = interpolate_3D_activation_map,
                        align_centers=True)
    

# %% Grad-CAM
debug_plot_grad_CAM = True
if debug_plot_grad_CAM:
    show_config = config.get('show', {})
    import numpy as np
    from utils.CAM_utils import get_target_sector
    from modules.visualization.pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
    # from utils.scar_utils import find_connected_components_binary_1d
    net.network.double()
    # net.network.float()
    layers = network.reg_layers_ModuleList
    # layers = network.joint_layers_ModuleList
    linear_layer_idx = np.where([type(layer) is torch.nn.modules.linear.Linear for layer in layers])[0]
    # if type(layers[linear_layer_idx[0] - 1]) is torch.nn.modules.flatten.Flatten:
    #     layer_before_first_linear = layers[linear_layer_idx[0] - 2]
    # else:
    #     layer_before_first_linear = layers[linear_layer_idx[0] - 1]
    conv_layers = [layer for layer in layers if type(layer) is torch.nn.modules.conv.Conv2d]
    linear_layers = [layer for layer in layers if type(layer) is torch.nn.modules.linear.Linear]
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
        # ic(target_category)
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
                                  input_info = config['data']['input_info'],
                                  output_info = config['data']['output_info'],
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
        # data_to_show = test_data[:4]
        # data_to_show = test_data[-4:]
        data_to_show = test_data[3:]
        # data_to_show = training_data_to_save[:4]
        
        # plot_multi_strainmat_with_curves_and_activation_map(
        #     data=data_to_show, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
        #     cam_data_types=[data_type['type'] for data_type in config['data']['output_info']] + ['total_loss'], counter_cam_data_types=[data_type['type'] for data_type in config['data']['output_info']], 
        #     fig=None, axs=None, 
        #     legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in data_to_show], 
        #     vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
        #     colors=None,
        #     check_activation_sectors = [datum['cam']['sector'] for datum in data_to_show])
        # curve_types_to_plot = []
        plot_multi_strainmat_with_curves_and_activation_map(
            data=data_to_show, strainmat_type=config['data']['input_info'][0]['type'], curve_types=curve_types_to_plot, 
            cam_data_types=[data_type['type'] for data_type in config['data']['output_info'] if data_type['tag'] == 'reg'], counter_cam_data_types=[data_type['type'] for data_type in config['data']['output_info'] if data_type['tag'] == 'reg'], 
            fig=None, axs=None, 
            legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in data_to_show], 
            vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, 
            colors=None,
            check_activation_sectors = [datum['cam']['sector'] for datum in data_to_show])
