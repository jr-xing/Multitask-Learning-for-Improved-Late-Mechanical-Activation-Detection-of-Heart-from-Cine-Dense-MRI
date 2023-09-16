# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:32:12 2021

@author: remus
"""
import nni
import argparse
import torch
import yaml, json
from pathlib import Path, PurePath
from datetime import date
import numpy as np
from utils.io import load_data_from_table


# from configs.getList import get_list
def run_experiment(config, NNI=False, logger=None, save_model=True, save_prediction=True, save_config=True,
                   trained_model_filename=None):
    # %% 0. Get experiment folder if needed
    trail_name = config['NNI'].get('trial_name',
                                   f"exp-{str(config['NNI']['experiment_id']).strip()}-idx-{config['NNI']['sequence_id']:03d}-trial-{str(config['NNI']['trial_id']).strip()}")

    # %% 1. Set device
    # gpuIdx = 0
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # %% 2. Load data
    if logger is not None:
        logger.info('Load Data')
    print('(PRINT) Load Data')

    # included_data_types = [input_type['type'] for input_type in config['data']['input_types'] + config['data']['output_types']]
    included_data_info = config['data']['input_info'] + config['data']['output_info']
    included_data_types = [data_type['type'] for data_type in included_data_info]
    # included_data_types = list(np.unique(config['data']['input_types'] + config['data']['output_types']))
    dataFilename = config['data']['filename']
    dataFull = load_data_from_table(dataFilename, data_types=included_data_info)
    # dataFull = np.load(dataFilename, allow_pickle=True).item()
    # dataInfo = dataSaved['description']
    # dataFull = dataSaved['data']
    if config['data'].get('scar_free', True):
        dataFull = [datum for datum in dataFull if datum['hasScar'] == 0]

    # Rule out problematic data
    # from utils.data import getPatientName, getSliceName
    # for datum in dataFull:
    #     datum['slice_name'] = getSliceName(datum['dataFilename'])
    #     datum['patient_name'] = getPatientName(datum['dataFilename'])
    #     datum['patient_slice_name'] = datum['patient_name'] + '-' + datum['slice_name']

    # included_data_types = list(np.unique(config['data']['input_types'] + config['data']['output_types'] + [config['data']['TOS_type']]))

    # problematic_GT_list = get_list(name = 'problematic_GT_2')        
    # problematic_GT_list = ['SET01-CT12_s']
    # dataFull = [datum for datum in dataFull if datum['patient_name'] not in problematic_GT_list]

    # Add class label or distance map
    from utils.data import add_classification_label, add_distancce_map
    # if config['eval']['method'] in ['multitask-reg-cls', 'multitask-reg-clsDistMap']:
    if config['eval']['method'] in ['multitask-reg-cls']:
        add_classification_label(dataFull, {}, included_data_info)

    if config['eval']['method'] in ['multitask-reg-clsDistMap']:
        # add_distancce_map(dataFull, config['eval']['paras'], config['data']['input_types'] + config['data']['output_types'])
        add_distancce_map(dataFull, included_data_info)

    # %% 3. Pre-processing
    # from utils.data import get_data_type_category
    # for data_type in config['data']['input_types']:
    #     if get_data_type_category(data_type) == 'strainmat':
    #         for datum in dataFull:
    #             datum[data_type] = np.flip(datum[data_type], axis=-2)

    from utils.preprocessing import unify_n_frame, unify_n_sector
    unify_n_frame(dataFull, included_data_types, n_frames_target='power_of_2')
    unify_n_sector(dataFull, included_data_types, n_sectors_target='power_of_2', method='copy_boundary')

    # %% 4. Train-test split
    from utils.data import train_test_split
    training_data_raw, test_data = train_test_split(config['data']['train_test_split'], dataFull)
    print('training_data_raw len:', len(training_data_raw))
    print('test_data len:', len(test_data))

    # %% 5. Augmentation
    for datum in dataFull:
        datum['augmented'] = False

    from utils.augmentation import augment
    # print('Augmentation:', config['data']['augmentation'])
    training_data_aug, training_data_aug_samples = augment(training_data_raw, included_data_types,
                                                           config['data']['augmentation'])
    print('training_data_aug len:', len(training_data_aug))
    print('training_data_aug_samples len:', len(training_data_aug_samples))

    # Update the distance map of augmented data
    # if config['eval']['method'] in ['multitask-reg-cls', 'multitask-reg-clsDistMap']:
    #     add_classification_label(training_data_aug + training_data_aug_samples, config['eval']['paras'], config['data']['input_types'] + config['data']['output_types'])
    # if config['eval']['method'] in ['multitask-reg-clsDistMap']:
    #     add_distancce_map(training_data_aug + training_data_aug_samples, config['eval']['paras'], config['data']['input_types'] + config['data']['output_types'])

    training_data = training_data_raw + training_data_aug

    # %% 5.1 Re-label for some types
    from utils.data import add_polyfit_coefficient
    for data_type in included_data_types:
        if data_type == 'polyfit_coefs':
            add_polyfit_coefficient(training_data + test_data, included_data_types, degree=10)

    # %% 6. Set Dataset
    from modules.dataset import Dataset
    dataset_precision = np.float16
    training_dataset = Dataset(training_data, config['data']['input_info'], config['data']['output_info'], precision=dataset_precision)
    test_dataset = Dataset(test_data, config['data']['input_info'], config['data']['output_info'], precision=dataset_precision)

    # %% 7. Set Network
    from utils.data import get_data_type
    if trained_model_filename is None:
        from modules.networks.get_network import get_network_by_name
        config['net']['paras']['n_sectors_in'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
        config['net']['paras']['n_sectors_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
        config['net']['paras']['n_frames'] = training_data[0][config['data']['input_info'][0]['type']].shape[-1]
        # config['net']['paras']['degree'] = 10
        if config['eval']['method'] in ['multitask-reg-cls']:
            cls_out_data_type = get_data_type('sector_label', config['data']['output_info'])
            config['net']['paras']['n_classes'] = training_data[0][cls_out_data_type].shape[-2]
        elif config['eval']['method'] in ['multitask-reg-clsDistMap']:
            cls_out_data_type = get_data_type('sector_dist_map', config['data']['output_info'])
            config['net']['paras']['n_classes'] = training_data[0][cls_out_data_type].shape[-2]
        # config['net']['paras']['n_sectors_out'] = 22
        config['net']['paras']['reg_n_dim_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2]
        config['net']['paras']['cls_n_dim_out'] = training_data[0][config['data']['input_info'][0]['type']].shape[-2] * \
                                                  training_data[0][cls_out_data_type].shape[-2]
        network = get_network_by_name(config['net']['type'], config['net'])
        network.set_input_types(training_dataset.input_types)
        network.set_output_types(training_dataset.output_types)
        network.to(device)
    else:
        network = torch.load(trained_model_filename)
    # print(network)

    # %% 8. Set Network Module
    from modules.net import NetModule
    net = NetModule(network=network, evaluation_config=config['eval'], regularization_config=config['regularization'],
                    device=device)

    # %% 9. Training
    if trained_model_filename is None:
        net.network.train()
        training_loss_final, training_loss_history, valid_loss_final, valid_loss_history, valid_reg_loss_final, past_time = \
            net.train(training_dataset=training_dataset, valid_dataset=test_dataset, training_config=config['training'],
                      NNI=NNI, logger=logger)
    else:
        training_loss_final, valid_loss_final = None, None

    # %% 10. Save Model
    # if save_model:
    #     save_model_filename = f"{trail_name}-network.pth"        
    #     save_model_filename_full = str(PurePath(config['exp_parent_path'], 'networks', save_model_filename))
    #     torch.save(network, save_model_filename_full)

    # %% 10. Save Prediction Results
    # Prediction
    net.network.eval()
    # training_data_to_save = training_data_raw[:12] + training_data_aug[:5] + training_data_aug_samples
    training_patients_to_show = ['SET01-CT02', 'SET01-CT16']
    training_data_to_save = training_data_raw[::5][:12] + [d for d in training_data_raw if d['patient_name'] in training_patients_to_show]
    for datum in training_data_raw + test_data + training_data_to_save:
        datum_pred = net.pred(datum, training_dataset.input_types, training_dataset.output_types)
        for output_data_key in datum_pred['data'].keys():
            datum[output_data_key + '_pred'] = datum_pred['data'][output_data_key]
        datum['loss_pred'] = datum_pred['loss']

    # expPath = '../nni_logs/' + config['NNI']['experiment_id'] + '/trials/' + config['NNI']['trial_id'] + '/'
    from utils.plot import save_multi_strainmat_with_curves
    curve_types_to_plot = []
    for output_data_type in [data_info['type'] for data_info in config['data']['output_info']]:
        curve_types_to_plot.append(output_data_type)
        curve_types_to_plot.append(output_data_type + '_pred')

    if save_prediction:
        save_test_results_filename_full = str(
            PurePath(config['exp_parent_path'], 'test_results', f"{trail_name}-test-results.pdf"))
        strainmat_type = config['data']['input_info'][0]['type']
        save_multi_strainmat_with_curves(data=test_data,
                                         strainmat_type=strainmat_type,
                                         curve_types=curve_types_to_plot,
                                         legends=curve_types_to_plot,
                                         save_filename=save_test_results_filename_full,
                                         subtitles=[datum['patient_slice_name'] for datum in test_data])

        save_training_results_filename_full = str(
            PurePath(config['exp_parent_path'], 'training_results', f"{trail_name}-train-results.pdf"))
        save_multi_strainmat_with_curves(data=training_data_to_save,
                                         strainmat_type=strainmat_type,
                                         curve_types=curve_types_to_plot,
                                         legends=curve_types_to_plot,
                                         save_filename=save_training_results_filename_full,
                                         subtitles=[datum['patient_slice_name'] for datum in training_data_to_save])
    debug_plot = False
    if debug_plot:
        from utils.plot import plot_multi_strainmat_with_curves
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(4, 4)
        axs = axs.flatten()
        plot_multi_strainmat_with_curves(data=test_data[:1],
                                         strainmat_type=config['data']['input_types'][0],
                                         curve_types=curve_types_to_plot,
                                         fig=fig, axs=axs,
                                         legends=curve_types_to_plot,
                                         subtitles=[datum['patient_slice_name'] for datum in test_data])

    # %% 11. Save Config
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

    return network, config['performance'], training_data_raw, test_data


# %%
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


import logging
from configs.getconfig import get_default_config, modify_config, update_iotype_argument
from utils.io import create_folder

_logger = logging.getLogger('mnist_example')
_logger.setLevel(logging.INFO)
from utils.augmentation import update_augment_argument

if __name__ == '__main__':
    tuned_params = nni.get_next_parameter()
    if tuned_params == {}:
        debug_mode = True
    else:
        debug_mode = False

    # exp_type = 'strainmat_to_TOS'
    # exp_type = 'multitask_reg_cls'
    # exp_type = 'multitask_reg_clsDistMap'
    # exp_type = 'multitask_reg_cls'
    exp_type = tuned_params['exp_type']
    config = get_default_config(exp_type)
    # config['training']['epochs_num'] = 50

    # Fetch hyper-parameters from HPO tuner
    # comment out following two lines to run the code without NNI framework


    modify_config(config, exp_type, tuned_params)

    if debug_mode:
        path_to_exp = 'D:\\Research\\Cardiac\\Experiment_Results\\NNI_test'
        # path_to_exp = '/home/jrxing/WorkSpace/Research/Cardiac/experiment_results/NNI_test'
        # path_to_exp = '../../experiment_results/NNI_test'
        config['exp_parent_path'] = path_to_exp
        create_folder(path_to_exp + '/training_results', recursive=False, action_when_exist='pass')
        create_folder(path_to_exp + '/test_results', recursive=False, action_when_exist='pass')
        create_folder(path_to_exp + '/networks', recursive=False, action_when_exist='pass')
        create_folder(path_to_exp + '/configs', recursive=False, action_when_exist='pass')
        config['training']['epochs_num'] = 71
        config['training']['batch_size'] = 32
        # config['data']['input_type'] = 'strainMat'
        # config['data']['output_type'] = 'TOS18_Jerry'

        config['data']['scar_free'] = False
        # config['data']['train_test_split']['paras']['test_patient_names'] = ['SET01-CT11', 'SET02-CT28', 'SET03-EC21',
        #                                                                      'SET01-CT02', 'SET01-CT16', 'SET01-CT18']


        # config['data']['filename'] = 'D://dataFull-201-2020-12-23-Jerry.npy'

        # config['data']['filename'] = 'D://dataFull-201-2020-12-23-Jerry.npy'
        # config['data']['filename'] = PurePath('/home/jrxing/WorkSpace/Research/Cardiac/Dataset', 'dataFull-201-2020-12-23-Jerry.npy')
        # config['data']['filename'] = str(PurePath('../../Dataset', 'dataFull-201-2020-12-23-Jerry.npy'))
        config['data']['input_info'] = update_iotype_argument('strainMatFullResolution')
        config['data']['TOS_info'] = update_iotype_argument('TOSfullRes_Jerry')
        # config['data']['output_types'] = 'TOSfullRes_Jerry+late_acti_label'.split('+')
        # config['data']['output_types'] = 'TOSfullRes_Jerry+late_acti_label'.split('+')
        # config['data']['output_types'] = 'TOSfullRes_Jerry+strain_curve_type_label'.split('+')
        # config['data']['output_info'] = update_iotype_argument('TOSfullRes_Jerry+scar-AHA-step=50')
        config['data']['output_info'] = update_iotype_argument('TOSfullRes_Jerry+scar-AHA-distmap')
        # config['data']['output_types'] = 'TOSfullRes_Jerry+strain_curve_type_dist_map'.split('+')
        # config['data']['output_types'] = 'polyfit_coefs+late_acti_label'.split('+')
        # config['data']['output_types'] = 'TOSfullRes_Jerry+late_acti_dist_map'.split('+')
        # config['data']['input_types'] = 'strainMat'.split('+')
        # config['data']['output_types'] = 'TOS18_Jerry+late_acti_label'.split('+')
        # config['data']['output_types'] = 'TOS18_Jerry+late_acti_dist_map'.split('+')
        config['data']['augmentation'] = update_augment_argument('shift-sector=-32_32_5+mixup=0.2_500', [])
        # config['data']['augmentation'] = update_augment_argument('shift-sector_-32_32_5', [])
        # print('AUG', update_augment_argument('shift-sector=-32_32_5+mixup=0.1_1000', []))
        logger = _logger
        # assert 1>10

        # config['net']['type'] = 'NetStrainMat2PolycoeffCls'
        config['net']['paras']['batch_norm'] = True
        config['eval']['paras']['cls_weight'] = 1e1


    else:
        exp_info_filename = PurePath(str(Path.home()), f'nni-config-{date.today().strftime("%Y-%m")}.yml')
        # exp_info_filename = f'./NNI/configs/nni-config-{date.today().strftime("%Y-%m")}.yml'
        with open(exp_info_filename) as file:
            exp_info = yaml.full_load(file)
        config['exp_parent_path'] = exp_info['exp_parent_path']
    # config['data']['filename'] = '../../Dataset/dataFull-201-2020-12-23-Jerry.npy'
    dataset_dir = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry'
    # data_records_filename = str(Path(dataset_dir, 'record_sheets\\cardiac-strainmat-dataset-2021-04-20.xlsx'))
    data_records_filename = str(Path(dataset_dir, 'record_sheets\\cardiac-strainmat-dataset-2021-04-20.xlsx'))
    config['data']['filename'] = data_records_filename
    config['data']['train_test_split']['paras']['test_patient_names'] = ['SET01-CT11', 'SET02-CT28', 'SET03-EC21',
                                                                         'SET03-UP34']

    config['NNI'] = {
        'experiment_id': nni.get_experiment_id(),
        'trial_id': nni.get_trial_id(),
        'sequence_id': nni.get_sequence_id()
    }
    config['NNI'][
        'trial_name'] = f"exp-{str(config['NNI']['experiment_id']).strip()}-idx-{config['NNI']['sequence_id']:03d}-trial-{str(config['NNI']['trial_id']).strip()}"

    _logger.info('Hyper-parameters: %s', config)

    # config['data']['train_test_split']['paras']['test_patient_names'] = ['SET01-CT11', 'SET02-CT28', 'SET03-EC21',
    #                                                                      'SET01-CT02', 'SET01-CT16', 'SET01-CT18']

    # Train network and getr performance
    network, performance, _, _ = run_experiment(config, NNI=True, logger=_logger)
    # network = None
    # config['NNI']['trial_id'] = 'SE'
    # performance = {'valid_loss': 20}

    # exp_trial_performance_log_new = {
    #     'exp_id': config_after_training['NNI']['experiment_id'], 'trial_id': config_after_training['NNI']['trial_id'],
    #     'training_loss': config_after_training['performance']['training_performance'],
    #     'valid_loss': config_after_training['performance']['valid_performance']
    #     }

    # Load performance log file
    # exp_trial_performance_log_filename = PurePath(config['exp_parent_path'], 'trial_performance_log.yml')
    exp_trial_performance_log_filename = config['exp_parent_path'] + 'trial_performance_log.yml'
    try:
        with open(exp_trial_performance_log_filename, 'r') as yamlfile:
            exp_trial_performance_log = yaml.safe_load(yamlfile)  # Note the safe_load
    except:
        exp_trial_performance_log = None

    if exp_trial_performance_log is None:
        exp_trial_performance_log = {config['NNI']['trial_id']: performance}
        trail_is_first = True
    else:
        exp_trial_performance_log[config['NNI']['trial_id']] = performance
        trail_is_first = False

    # Save network model if current performance is better than the best one
    if trail_is_first:
        save_model = True
    else:
        valid_performance_log = [exp_trial_performance_log[trial_id]['dafault_loss'] for trial_id in
                                 exp_trial_performance_log.keys()]
        if performance['dafault_loss'] <= min(valid_performance_log):
            save_model = True
        else:
            save_model = False

    if save_model:
        save_model_filename = f"{config['NNI']['trial_name']}-network.pth"
        save_model_filename_full = str(PurePath(config['exp_parent_path'], 'networks', save_model_filename))
        torch.save(network, save_model_filename_full)

    # Save current performance
    with open(exp_trial_performance_log_filename, 'w') as yamlfile:
        yaml.safe_dump(exp_trial_performance_log, yamlfile)  # Also note the safe_dump
