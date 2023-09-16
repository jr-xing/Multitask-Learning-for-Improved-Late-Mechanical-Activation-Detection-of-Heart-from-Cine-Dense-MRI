# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:58:59 2021

@author: Jerry Xing
"""

#%% 1. Set config and saved model filename
network_filename = 'D:\\Research\\Cardiac\\Experiment_Results\\2021-03-09-multitask_reg_cls\\networks\\exp-m5FMd48l-idx-017-trial-TvcFS-network.pth'
config_filename = 'D:\\Research\\Cardiac\\Experiment_Results\\2021-03-09-multitask_reg_cls\\configs\\exp-m5FMd48l-idx-017-trial-TvcFS-config.json'

# network_filename = 'D:\\Research\\Cardiac\\Experiment_Results\\2021-02-20-multitask_reg_cls\\networks\\exp-e9ToIRxp-idx-002-trial-tYKlw-network.pth'
# config_filename = 'D:\\Research\\Cardiac\\Experiment_Results\\2021-02-20-multitask_reg_cls\\configs\\exp-e9ToIRxp-idx-002-trial-tYKlw-config.json'
# network_filename = './temp/exp-2aFH3Zfi-idx-002-trial-Cmgqn-network.pth'
# config_filename = './temp/exp-2aFH3Zfi-idx-002-trial-Cmgqn-config.json'

# network_filename = './temp/exp-incredible-STANDALONE-idx-000-trial-STANDALONE-network.pth'
# config_filename = './temp/exp-incredible-STANDALONE-idx-000-trial-STANDALONE-config.json'

#%% 2. Load config
import json
with open(config_filename, 'r') as file:
    config = json.load(file)

#%% 3. Modify config
config['data']['scar_free'] = True
config['data']['filename'] = 'D://dataFull-201-2020-12-23-Jerry.npy'

#%% 3. Get Predicrtion
# Load network
if False:
    from run_exp_func import run_experiment
    network, _, training_data_raw, test_data = run_experiment(config, NNI=False, logger = None, 
                                                              save_model = False, save_prediction = False, save_config = False, 
                                                              trained_model_filename = network_filename)
    
    from utils.plot import plot_strainmat_with_curves
    training_data_aug = [datum for datum in training_data_raw if datum['augmented'] == True]
    plot_strainmat_with_curves(training_data_aug[0]['strainMatFullResolution'], [training_data_aug[0]['TOSfullRes_Jerry']], ['TOSfullRes_Jerry'])

#%%
import torch
network = torch.load(network_filename)

#%% 4. Data to show
import numpy as np

ISMRM_data_filename = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\paper\\MICCAI_2021\\code_from_ISMRM_2021\\prediction_from_ISMRM_include_scar.npy'
# ISMRM_data_filename = './prediction_from_ISMRM_include_scar.npy'
ISMRM_data = np.load(ISMRM_data_filename, allow_pickle = True).item()


# scar_data = [datum for datum in training_data_raw + test_data if datum['hasScar']]
# data_to_show = test_data + scar_data
# from utils.plot import plot_multi_strainmat_with_curves
# # plot_multi_strainmat_with_curves
# n_plot_per_fig = 12
# curve_types_to_plot = []
# for output_data_type in config['data']['output_types']:
#     curve_types_to_plot.append(output_data_type)
#     curve_types_to_plot.append(output_data_type + '_pred')
# for fig_idx in range(len(data_to_show)//n_plot_per_fig + 1):
#     data_to_show_curr_fig = data_to_show[fig_idx*n_plot_per_fig : (fig_idx + 1)*n_plot_per_fig]
#     plot_multi_strainmat_with_curves(data_to_show_curr_fig, 
#                                      strainmat_type = config['data']['input_types'][0], curve_types = curve_types_to_plot, legends = curve_types_to_plot, 
#                                      n_cols=4,subtitles=[datum['patinent_slice_name'] for datum in data_to_show_curr_fig])



#%% Load simple regression network output from ISMRM

from utils.preprocessing import unify_n_sector
device = torch.device('cuda:0')

unify_n_sector(ISMRM_data['dataTe'] + ISMRM_data['dataTr'], config['data']['input_types'] + config['data']['output_types'] + ['activeContourResultFullRes', 'pred'], n_sectors_target = 'power_of_2', method = 'copy_boundary')

from utils.data import getPatientName, getSliceName
for datum in ISMRM_data['dataTe'] + ISMRM_data['dataTr']:
    datum['patient_name'] = getPatientName(datum['dataFilename'])
    datum['slice_name'] = getSliceName(datum['dataFilename'])
    datum['patient_slice_name'] = getPatientName(datum['dataFilename']) + '-' + getSliceName(datum['dataFilename'])

# problmatic_data = ['SET01-CT11-SL1', 'SET01-CT11-SL2']
# ISMRM_data['dataTe'] = [datum for datum in ISMRM_data['dataTe'] if datum['patient_slice_name'] not in problmatic_data]
#%%
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('%matplotlib auto')

#%%
import time
multitask_pred_times = []

joint_net_reg_loss = []
joint_net_reg_masked_loss = []
simple_net_reg_loss = []
ac_reg_loss = []

for datum in ISMRM_data['dataTe'] + ISMRM_data['dataTr']:
    mat = datum['strainMatFullResolution']    
    new_multitask_network_regression_input = {'strainMatFullResolution': torch.flip(torch.from_numpy(mat).to(device, dtype = torch.float), dims=[-2])}
    # new_multitask_network_regression_input = {'strainMatFullResolution': torch.from_numpy(mat).to(device, dtype = torch.float)}
    # new_multitask_network_regression_input['strainMatFullResolution'] = torch.roll(new_multitask_network_regression_input['strainMatFullResolution'], 30, -2)*0
    
    # new_multitask_network_regression_input = {'strainMatFullResolution': torch.rand(1,1,128,64).to(device, dtype = torch.float)}
    multitask_start_time = time.time()
    new_multitask_network_regression_output = network(new_multitask_network_regression_input)
    multitask_pred_times.append(time.time() - multitask_start_time)
    new_multitask_network_regression_output_reg = new_multitask_network_regression_output['TOSfullRes_Jerry'].cpu().detach().numpy()
    # new_multitask_network_regression_output_cls = new_multitask_network_regression_output['late_acti_label'].cpu().detach().numpy()[0,1,:] > 0.999
    datum['pred_new_multitask'] = new_multitask_network_regression_output_reg
    
    # datum['late_acti_label'] = new_multitask_network_regression_output_cls[None,:]
    # datum['pred_new_multitask_masked'] = new_multitask_network_regression_output_reg.copy()
    # datum['pred_new_multitask_masked'][datum['late_acti_label'] < 0.1] = 17
    
    joint_net_reg_loss.append(np.linalg.norm(new_multitask_network_regression_output_reg - datum['TOSfullRes_Jerry']))
    # joint_net_reg_masked_loss.append(np.linalg.norm(datum['pred_new_multitask_masked'] - datum['TOSfullRes_Jerry']))
    simple_net_reg_loss.append(np.linalg.norm(datum['pred'] - datum['TOSfullRes_Jerry']))
    ac_reg_loss.append(np.linalg.norm(datum['activeContourResultFullRes'] - datum['TOSfullRes_Jerry']))
    # print(datum.keys())
# new_multitask_regression = [netrowk()]

joint_reg_mean_loss = np.mean(joint_net_reg_loss)
# joint_reg_masked_mean_loss = np.mean(joint_net_reg_masked_loss)
simple_net_reg_mean_loss = np.mean(simple_net_reg_loss)
ac_reg_mean_loss = np.mean(ac_reg_loss)

#%%
# import matplotlib.pyplot as plt
# dd = datum['strainMatFullResolution']
# ddroll = np.roll(datum['strainMatFullResolution'], -30, -2)
# plt.figure();plt.pcolor(np.squeeze(dd), cmap = 'jet', vmax = 0.2, vmin = -0.2)
# plt.figure();plt.pcolor(np.squeeze(ddroll), cmap = 'jet', vmax = 0.2, vmin = -0.2)

#%%
import matplotlib.pyplot as plt
boxprops = dict(linestyle='-', linewidth=3, color='black')
medianprops = dict(linestyle='-', linewidth=3, color='orange')
whiskerprops = dict(linestyle='-', linewidth=2, color='black')

fig_boxplot, ax_plot = plt.subplots()
# ax_plot.boxplot([joint_net_reg_loss, simple_net_reg_loss, ac_reg_loss], usermedians = [joint_reg_mean_loss, simple_net_reg_mean_loss, ac_reg_mean_loss])
ax_plot.boxplot([joint_net_reg_loss, simple_net_reg_loss, ac_reg_loss],
                boxprops=boxprops, medianprops=medianprops,whiskerprops=whiskerprops,capprops=whiskerprops)
ax_plot.set_xticklabels(['Our method', 'Reg Network', 'Active contour'], fontsize=18)
ax_plot.set_ylabel('L2 Loss', fontsize=18)
    
fig_boxplot, ax_plot = plt.subplots()
# ax_plot.boxplot([joint_net_reg_loss, simple_net_reg_loss], usermedians = [joint_reg_mean_loss, simple_net_reg_mean_loss])
ax_plot.boxplot([joint_net_reg_loss, simple_net_reg_loss],
                boxprops=boxprops, medianprops=medianprops,whiskerprops=whiskerprops,capprops=whiskerprops)
ax_plot.set_xticklabels(['Our method', 'Reg Network'])
ax_plot.set_ylabel('L2 Loss')
#%%
import matplotlib.pyplot as plt
# data_to_plot = ISMRM_data['dataTe'] + ISMRM_data['dataWithScar']
data_to_plot = ISMRM_data['dataTe']
# curve_types_to_plot = ['pred_new_multitask', 'TOSfullRes_Jerry', 'pred', 'pred_pred']
# curve_types_to_plot = ['TOSfullRes_Jerry', 'activeContourResultFullRes', 'pred', 'pred_new_multitask', 'late_acti_label']
# legends = ['Manual label', 'Active contour', 'Reg network', 'Our method', 'cls']
# colors = ['k', 'green', 'blue', 'orange', 'red']

curve_types_to_plot = ['TOSfullRes_Jerry', 'activeContourResultFullRes', 'pred', 'pred_new_multitask']
legends = ['Manual label', 'Active contour', 'Reg network', 'Our method', 'cls']
colors = ['k', 'green', 'blue', 'orange', 'red']

# curve_types_to_plot = ['late_acti_label']
# legends = ['cls']
# colors = ['red']

n_cols = 4
n_rows = 4
n_figs = len(data_to_plot) // (n_cols*n_rows) + 1

from utils.plot import plot_multi_strainmat_with_curves
for fig_idx in range(n_figs):
    data_curr_fig = data_to_plot[fig_idx*(n_cols*n_rows): min((fig_idx+1)*(n_cols*n_rows), len(data_to_plot))]    
    
    fig, axs = plt.subplots(n_rows, n_cols)
    axs = axs.flatten()
    plot_multi_strainmat_with_curves(data = data_curr_fig, 
        strainmat_type = config['data']['input_types'][0], 
        curve_types = curve_types_to_plot,
        fig = fig, axs = axs,
        legends = legends,
        subtitles=[datum['patient_slice_name'] for datum in data_curr_fig], flipStrainMat = True, colors = colors)

# data_to_show_names = [['SET01', 'CT11', 'SL7'],
#                       ['SET02', 'CT28', 'SL2'],
#                       ['SET02', 'CT28', 'SL5'],
#                       ['SET02', 'CT28', 'SL6']]
#%%
# curve_types_to_plot = ['TOSfullRes_Jerry', 'activeContourResultFullRes', 'pred', 'pred_new_multitask']
curve_types_to_plot = ['TOSfullRes_Jerry', 'activeContourResultFullRes', 'pred', 'pred_new_multitask']
# legends = ['Manual label', 'Active contour', 'Reg network', 'Our method', 'cls']
legends = ['Manual label', 'Active contour', 'Reg network', 'Our method', 'cls']
colors = ['k', 'green', 'blue', 'orange', 'red']

data_to_show_names = [['SET01', 'CT11', 'SL2'],
                      ['SET01', 'CT11', 'SL9'],
                      ['SET02', 'CT28', 'SL1'],
                      ['SET02', 'CT28', 'SL6']]

from utils.plot import plot_strainmat_with_curves
for datum in ISMRM_data['dataTe']:
    for data_to_show_name in data_to_show_names:
        if all(name in datum['dataFilename']for name in data_to_show_name):
            fig, axe = plt.subplots()
            plot_strainmat_with_curves(strainMat = datum[config['data']['input_types'][0]], 
                                       # curves = [datum['TOSfullRes_Jerry'], datum['activeContourResultFullRes'],datum['pred'], datum['pred_new_multitask']], 
                                       curves = [datum['TOSfullRes_Jerry'], datum['activeContourResultFullRes'],datum['pred']], 
                                       curve_types = curve_types_to_plot,
                                       axe = axe, legends = legends,title = None,
                                       colors = colors,
                                       flipTOS = True, flipStrainMat=True)
            # save_filename = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\paper\\MICCAI_2021\\figures\\experiment\\' + \
            #      'retrain-' + '-'.join(data_to_show_name) + '.pdf'
            # fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
            # plt.close(fig)
            continue


#%% For presentation
dd = [datum for datum in data_to_plot if 'CT11-SL1' in datum['patient_slice_name']][0]
fig, axe = plot_strainmat_with_curves(strainMat = dd[config['data']['input_types'][0]], 
                                       # curves = [datum['TOSfullRes_Jerry'], datum['activeContourResultFullRes'],datum['pred'], datum['pred_new_multitask']], 
                                       # curves = [dd['TOSfullRes_Jerry'], dd['activeContourResultFullRes'],dd['pred']], 
                                       curves = [dd['TOSfullRes_Jerry'], dd['activeContourResultFullRes']], 
                                       curve_types = curve_types_to_plot,
                                       axe = None, legends = legends,title = None,
                                       colors = colors,
                                       flipTOS = True, flipStrainMat=True)

axe.legend(prop={'size': 20})
#%%
# if False:
if True:
    from utils.data import getPatientName
    from utils.TOS3DPlotInterpFunc import TOS3DPlotInterp
    dataTe_patientNames = np.unique([getPatientName(datum['dataFilename']) for datum in ISMRM_data['dataTe']])
    dataTe_to_show_3D = [datum for datum in ISMRM_data['dataTe'] if getPatientName(datum['dataFilename']) == dataTe_patientNames[1]]
    # dataTe_to_show_3D = [dataTe_to_show_3D[idx] for idx in [0,1,2,4]]
    # dataTe_to_show_3D = [dataTe_to_show_3D[idx] for idx in [0,1,2,6]]
    # dataTe_to_show_3D = [dataTe_to_show_3D[idx] for idx in [0,1,2,6]]
    # dataTe_to_show_3D = [dataTe_to_show_3D[idx] for idx in [0,1,2,3,4,5,6]]
    dataTe_to_show_3D = [dataTe_to_show_3D[idx] for idx in [0,1,2,4,5,6]]
    # dataTr_patientNames = np.unique([getPatientName(datum['dataFilename']) for datum in dataTrRaw])[1:4]
    # dataTr_to_show_3D = [datum for datum in dataTrRaw if getPatientName(datum['dataFilename']) == dataTr_patientNames[0]]
    restoreOriSlices = True
    vmax = 120
    # vmax = None
    
    TOS3DPlotInterp(dataTe_to_show_3D, tos_key = config['data']['output_types'][0], title = dataTe_patientNames[1] + '-GT', restoreOriSlices = restoreOriSlices, vmax = vmax)
    TOS3DPlotInterp(dataTe_to_show_3D, tos_key = 'activeContourResultFullRes', title = dataTe_patientNames[1] + '-active contour', restoreOriSlices = restoreOriSlices, vmax = vmax)
    TOS3DPlotInterp(dataTe_to_show_3D, tos_key = 'pred', title = dataTe_patientNames[1] + '-simple reg', restoreOriSlices = restoreOriSlices, vmax = vmax)
    TOS3DPlotInterp(dataTe_to_show_3D, tos_key = 'pred_new_multitask', title = dataTe_patientNames[1] + '-multitask reg', restoreOriSlices = restoreOriSlices, vmax = vmax)

#%% Load new segmentation from Mohamad
import scipy.io as sio
mat_with_new_seg_filename = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET02\\CT28\\LGEs\\SA_rev.mat'

mat_with_new_seg = sio.loadmat(mat_with_new_seg_filename, struct_as_record=False, squeeze_me = True)



#%%    
for axe_idx, (tos_key, tos_name) in enumerate(zip([config['data']['output_types'][0], 'activeContourResultFullRes', 'pred', 'pred_new_multitask_masked'], ['GT', 'AC', 'Reg', 'Multitask'])):
    fig = plt.figure()
    axe = fig.gca(projection='3d')
    TOS3DPlotInterp(dataTe_to_show_3D, tos_key = tos_key, title = None, restoreOriSlices = restoreOriSlices, vmax = vmax, axe = axe)
    save_filename = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\paper\\MICCAI_2021\\figures\\experiment\\' + \
                    'retrain-3DActiMap-' + tos_name + \
                    '.pdf'
    fig.savefig(save_filename, bbox_inches='tight',transparent=True)   # save the figure to file    
    plt.close(fig)
#%%
import numpy as np
mat = np.squeeze(ISMRM_data['dataTe'][0]['strainMatFullResolution'])
TOS = np.squeeze(ISMRM_data['dataTe'][0]['late_acti_label'])
mat = np.flip(mat, axis=-2)

roll_amount = 10
mat_roll = np.roll(mat, roll_amount, axis=-2)
TOS_roll = np.roll(TOS, roll_amount, axis=-1)

import matplotlib.pyplot as plt
plt.figure();
# plt.pcolor(np.flipud(mat), cmap = 'jet', vmax = 0.2, vmin = -0.2)
plt.pcolor(mat, cmap = 'jet', vmax = 0.2, vmin = -0.2)
plt.plot(TOS.flatten()*17, np.arange(128))
plt.figure();
# plt.pcolor(np.flipud(mat_roll), cmap = 'jet', vmax = 0.2, vmin = -0.2)
plt.pcolor(mat_roll, cmap = 'jet', vmax = 0.2, vmin = -0.2)
plt.plot(TOS_roll.flatten()/17, np.arange(128))
# plt.figure();plt.pcolor(np.squeeze(ddroll), cmap = 'jet', vmax = 0.2, vmin = -0.2)

#%% 
running_time_AC_18 = [1.0052,0.5977,0.7817,0.8403,0.8172,0.6463,0.0209,0.0212,0.1425,0.3280,0.4313,0.7852,0.3303,0.3584,0.4668,0.2907,0.5398,0.5442,0.2706,0.5194,0.9269,1.0011,0.9836,0.9698,0.7248,0.9887,0.9687,0.9643,0.9639,0.6087,0.7555,0.5519,0.4647,0.3587,0.7753,0.4645,0.3922,0.3200,0.6515,0.5306,0.1921,0.6779,0.7390,0.3218,0.4869,0.7468,0.7514,0.7553,0.7425,0.7502,0.7497,0.7660,0.7731,0.6990,0.7585,0.7631,0.7610,0.7452,0.7564,0.7585,0.7381,0.7397,0.7395,0.3658,0.4265,0.7840,0.4193,0.7561,0.3342,0.7431,0.2223,0.4658,0.5122,0.5578,0.7646,0.5343,0.7634,0.3438,0.6223,0.2368,0.7549,0.5088,0.4106,0.7532,0.3905,0.5027,0.3878,0.5468,0.7372,0.8227,0.5760,0.7221,0.7479,0.4439,0.5104,0.7535,0.2512,0.2727,0.7474,0.7588,0.7630,0.3198,0.3398,0.0199,0.6716,0.7632,0.7588,0.4106,0.7562,0.7658,0.7637,0.7585,0.7514,0.7572,0.5726,0.3139,0.7440,0.8553,0.5159,0.8061,0.5531,0.4008,0.7592,0.8157,0.9434,0.8617,0.7618,0.1960]