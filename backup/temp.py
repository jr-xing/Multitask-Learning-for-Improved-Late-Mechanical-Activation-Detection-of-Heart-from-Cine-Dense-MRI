# %% Load TOS from TOS file
# TEst rclone
from pathlib import Path
import scipy.io as sio
import numpy as np
TOS_filename_1 = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT01\\TOS\\Base_mechDelay.mat'
TOS_filename_2 = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\Pre_CRT_LBBB_with_scar\\34_CM_MR\\TOS\\auto.1_Base_tos.mat'
TOS_mat = sio.loadmat(TOS_filename_2)
# def load_TOS(filename):
#     pass

def ensure_minimum_equals_17(TOS):
    TOS[TOS < 17] = 17
    return TOS

TOS_data = {'18': None, '126': None}
# Load TOS 18
TOS_18_keys_in_order = ['xs18_new', 'xs']
for TOS_18_key in TOS_18_keys_in_order:
    if TOS_18_key in TOS_mat.keys():
        TOS_data['18'] = ensure_minimum_equals_17(TOS_mat[TOS_18_key])
        break

# Load TOS 126
TOS_126_keys_in_order = ['xsfullRes_new', 'xsfullRes']
for TOS_126_key in TOS_126_keys_in_order:
    if TOS_126_key in TOS_mat.keys():
        TOS_data['126'] = ensure_minimum_equals_17(TOS_mat[TOS_126_key])
        break

# %% Load scar data
import pandas as pd
scar_filename = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\Pre_CRT_LBBB_with_scar\\34_CM_MR\\scar\\auto.1_Base.mat.csv'
scar_df = pd.read_csv(scar_filename)
scar_percentage = scar_df['scar_percentage'].to_numpy()

# %% Printing all slice names in Pre-CRT folder
import os, sys
from pathlib import PurePath
data = []
base_path = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\Pre_CRT_LBBB_with_scar\\'
patient_name_total_length = 20
for patient_name in os.listdir(base_path):
    mat_folder = PurePath(base_path, patient_name, 'mat')
    for mat_filename in os.listdir(mat_folder):
        slice_name = mat_filename.replace('.mat', '')
        space_len = patient_name_total_length - len(patient_name)
        # data.append({'patient': patient_name, 'slice': slice_name})
        print(patient_name + ' '*space_len + slice_name)
# for path, subdirs, files in os.walk(base_path):
#     for name in files:
#         print(os.path.join(path, name))

# %% Flip sector scar annotation
import pandas as pd
scar_file_path = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT02\\scar\\'
for scar_filename in os.listdir(scar_file_path):
    # print(scar_filename)
    scar_datum = pd.read_csv(scar_file_path + scar_filename)
    for key in ['scar_area', 'myocardium_area', 'scar_percentage']:
        print(scar_datum[key].to_numpy())
        scar_datum[key] = scar_datum[key].values[::-1]
        print(scar_datum[key].to_numpy())
    scar_datum.to_csv(scar_file_path + scar_filename)
    
# %% Transform AHA scar annotation to sector scar annotation
# Load strainmat files
import os
import scipy.io as sio
segment_6_name_in_order = ['Inferoseptum', 'Inferior', 'Inferolateral', 'Anterolateral', 'Anterior', 'Anteroseptum']
segment_4_name_in_order = ['Septum', 'Inferior', 'Lateral', 'Anterior']
strainmat_folder = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT16\\mat\\'
strainmat_data = []
for strainmat_filename in os.listdir(strainmat_folder):
    strainmat_mat = sio.loadmat(strainmat_folder + strainmat_filename)
    spl = strainmat_mat['SequenceInfo'][0,0]['SliceLocation'][0,0]
    strainmat_data.append((strainmat_filename, spl))

# Load AHA scar annotation
import pandas as pd
import numpy as np
AHA_file = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT16\\Scar-Annotation-AHA\\CT16-Late Enhancement - Polar Plot Summary Table.csv'
AHA_data = pd.read_csv(AHA_file)
AHA_unique_slice_names = list(np.unique([row['Label'].split(':')[0] for idx, row in AHA_data.iterrows()]))
     # 1		patient_name	slice_name	scar_area	myocardium_area	scar_percentage

scar_dict = {
    'patient_name': '',
    'sclie_name': '',
    'scar_area': np.zeros(126),
    'myocardium_area': np.zeros(126),
    'scar_percentage': np.zeros(126)}
for AHA_unique_slice_name in AHA_unique_slice_names:
    AHA_slice_rows = AHA_data[AHA_data['Label'].str.startswith(AHA_unique_slice_name)]    
    if len(AHA_slice_rows) == 6:
        for sector_name in segment_6_name_in_order:
            sector_row = AHA_slice_rows[AHA_slice_rows['Label'].str.contains(sector_name)]
    elif len(AHA_slice_rows) == 4:
        for sector_name in segment_4_name_in_order:
            sector_row = AHA_slice_rows[AHA_slice_rows['Label'].str.contains(sector_name)]
    else:
        raise ValueError("!")

# %% AHA array to scar label
import numpy as np
AHA_array = np.array([1,2,3,4])
n_target_sector = 126
segment_sector_len = n_target_sector // len(AHA_array)
if len(AHA_array) == 4:
    
    pass
elif len(AHA_array) == 6:
    pass

# %% Check # of data with scar
import numpy as np
from utils.io import load_data_from_table
dataFull = load_data_from_table(data_info=['strainMatFullResolution', 'scar_sector_label'])


print('# of patients', len(np.unique([datum['patient_name'] for datum in dataFull])))
print('# of slices', len([datum for datum in dataFull]))
print('# of patients in Pre-CRT with LBBB', len(np.unique([datum['patient_name'] for datum in dataFull if datum['patient_name'].startswith('Pre_CRT_LBBB_with_scar')])))
print('# of slice in Pre-CRT with LBBB', len([datum for datum in dataFull if datum['patient_name'].startswith('Pre_CRT_LBBB_with_scar')]))
print('# of LBBB patients with scar annoattion data: ', len(np.unique([datum['patient_name'] for datum in dataFull if 'scar_sector_percentage' in datum.keys()])))
print('# of slices with scar annoattion data: ', len([datum for datum in dataFull if 'scar_sector_percentage' in datum.keys()]))
print('# of slices with VALID scar annoattion data: ', len([datum for datum in dataFull if 'scar_sector_percentage' in datum.keys() and np.sum(datum['scar_sector_percentage']>0.2)>1]))

# %% Test Grad-CAM
# https://github.com/jacobgil/pytorch-grad-cam
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50

# model = resnet50(pretrained=True)
# target_layer = model.layer4[-1]
# input_tensor = # Create an input tensor image for your model..
# # Note: input_tensor can be a batch tensor with several images!

# # Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.use_cuda)

# # If target_category is None, the highest scoring category
# # will be used for every image in the batch.
# # target_category can also be an integer, or a list of different integers
# # for every image in the batch.
# target_category = 281

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]

# visualization = show_cam_on_image(rgb_img, grayscale_cam)

# %% Find center of scar region
import numpy as np
def find_connected_components_binary_1d(arr_bin):
    # return: an array, 0 -> background, 1,2,3,... different region labels
    arr_bin_padded = np.concatenate((np.zeros(1), arr_bin))
    region_label = np.zeros_like(arr_bin)
    curr_region_idx = 0
    is_inside_region = False
    for idx in range(len(arr_bin)):
        # up stair -> new region
        if arr_bin_padded[idx] == 0 and arr_bin_padded[idx + 1] == 1:
            is_inside_region = True
            curr_region_idx += 1
                
        # down stair -> end of region
        if arr_bin_padded[idx] == 1 and arr_bin_padded[idx + 1] == 0:
            is_inside_region = False        
        
        if is_inside_region:
            region_label[idx] = curr_region_idx
    
    regions = []
    for region_idx in range(1, curr_region_idx + 1):
        region_center = int(np.mean(np.where(region_label == region_idx)))
        region_length = np.sum(region_label == region_idx)
        regions.append({
            'center': region_center,
            'length': region_length
            })
    
    return regions, region_label

arr_bin = np.zeros(126)
arr_bin[10:20] = 1
arr_bin[70:90] = 1
regions, region_label = find_connected_components_binary_1d(arr_bin)

# %% Add binary mask
import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
img = rgb2gray(imread('./utils/saga.png'))
fig, axe = plt.subplots();
# axe.imshow(img, cmap='jet')
mask = np.zeros_like(img)
H, W = img.shape
for y in range(H):
    for x in range(W):
        mask[y, x] = (y-H/2)**2 + (x-W/2)**2
mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
# axe.imshow(np.zeros_like(img), alpha=np.maximum(mask, 0.5), cmap='gray', vmin = 0, vmax = 1)

def plot_activation_map(activation_map, strainmat=None, axe=None):
    # Create new fig if not provided
    if axe is None:
        fig, axe = plt.subplots()
    
    # Plot strain matrix
    if strainmat is None:
        strainmat = np.ones_like(activation_map)
        strainmat_vmax = 1
        strainmat_vmin = 0
    else:
        strainmat_vmax = 0.2
        strainmat_vmin = -0.2    
    axe.pcolor(strainmat, vmax = strainmat_vmax, vmin = strainmat_vmin, cmap = 'jet')
    
    # Plot activation map
    if np.max(activation_map) > 1e-5:
        activation_map_norm = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map))
    else:
        activation_map_norm = activation_map
    axe.pcolor(np.zeros_like(activation_map_norm), alpha=np.maximum(activation_map_norm, 0.5), cmap='gray', vmin = 0, vmax = 1)
    
plot_activation_map(mask, img, axe)

# %% Merge gaps
import numpy as np
from scipy.ndimage import binary_closing
import matplotlib.pyplot as plt
arr = np.zeros(40)
arr[5:10] = 1
arr[11:15] = 1
arr[17:20] = 1
arr[23:25] = 1
arr[30:35] = 1
fig, axs = plt.subplots(1,2)
axs[0].plot(arr)

arr_closed = binary_closing(arr, structure=np.ones((6)))
axs[1].plot(arr_closed)

from utils.io import load_data_from_table
input_info = [{'type': 'strainMatFullResolutionSVD', 'config':{}}]
data = load_data_from_table(dataset_dir=None,data_records_filename='D:\\Documents\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\record_sheets\\cardiac-strainmat-dataset-2021-06-13-scar-classification.xlsx', 
                            load_DENSE=True, load_TOS=False, load_scar=True, data_info=input_info)

for datum in data:
    if '121' in datum['patient_name']:
        print(datum.keys())

data_with_scar = [datum for datum in data if 'scar_sector_percentage' in datum.keys()]

scar_label_thres = 0.5
for datum in data_with_scar:    
    scar_sector_label = np.zeros(len(datum['scar_sector_percentage']))
    scar_sector_label[datum['scar_sector_percentage'] >= scar_label_thres] = 1
    datum['scar_sector_label'] = scar_sector_label

from utils.scar_utils import find_connected_components_binary_1d
def has_spike(binary_array, len_thres=5):
    if np.sum(binary_array) <= 1:
        return False
    connected_regions, _ = find_connected_components_binary_1d(binary_array, order='size')
    minimal_connected_region_len = connected_regions[-1]['length']
    if minimal_connected_region_len <= len_thres:
        return True
    else:
        return False

def has_gap(binary_array, len_thres=5):
    return has_spike(1 - binary_array, len_thres)
# datum_with_scar_spikes = [datum for datum in data_with_scar if has_scar_spike(datum['scar_sector_label'])]
    
from utils.scar_utils import remove_sector_label_spkies
for datum in data_with_scar:
    datum['scar_sector_label_connected'] = remove_sector_label_spkies(datum['scar_sector_label'] > 0, connect_len=20)
    
    # if has_spike(datum['scar_sector_label']) or has_gap(datum['scar_sector_label']):
    #     scar_sector_label = datum['scar_sector_label']
    #     scar_sector_label_repeated = np.tile(scar_sector_label, 3)
    #     scar_sector_label_repeated_closed = binary_closing(scar_sector_label_repeated, structure=np.ones((10)))
    #     datum['scar_sector_label_connected'] = scar_sector_label_repeated_closed[len(scar_sector_label):2*len(scar_sector_label)]
    #     # datum['scar_sector_label_connected'] = binary_closing(datum['scar_sector_label'], structure=np.ones((6)))
    # else:
    #     datum['scar_sector_label_connected'] = datum['scar_sector_label']
# aaa = np.array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.,
#      0. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0.,
#      0. , 0. , 0. , 0. , 1. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0.,
#      0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.,
#      0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.,
#      0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.])


n_plots = np.ceil(len(data_with_scar)/8).astype(int)
n_rows = 4
n_cols = 6
data_per_row = 2
plots_per_data = 3
for plot_idx in range(n_plots):
    fig, axs = plt.subplots(n_rows, n_cols)
    # axs = axs.flatten()
    for row_idx in range(n_rows):
        datum0_idx = plot_idx*n_rows*data_per_row + row_idx*data_per_row + 0
        datum1_idx = plot_idx*n_rows*data_per_row + row_idx*data_per_row + 1
        print(datum0_idx)
        print(datum1_idx)
        
        if datum0_idx > len(data_with_scar) - 1:
            break        
        
        datum0 = data_with_scar[datum0_idx]
        axs[row_idx, 0].plot(datum0['scar_sector_percentage']*10, np.arange(len(datum0['scar_sector_label'])), linewidth=2)
        axs[row_idx, 1].plot(datum0['scar_sector_label']*10, np.arange(len(datum0['scar_sector_label'])), linewidth=2)
        axs[row_idx, 2].plot(datum0['scar_sector_label_connected']*10, np.arange(len(datum0['scar_sector_label'])), linewidth=2)
        
        axs[row_idx, 0].pcolor(np.squeeze(datum0['strainMatFullResolutionSVD'][...,:-5]), vmax = 0.2, vmin = -0.2, cmap='jet')
        axs[row_idx, 1].pcolor(np.squeeze(datum0['strainMatFullResolutionSVD'][...,:-5]), vmax = 0.2, vmin = -0.2, cmap='jet')
        axs[row_idx, 2].pcolor(np.squeeze(datum0['strainMatFullResolutionSVD'][...,:-5]), vmax = 0.2, vmin = -0.2, cmap='jet')
        
        axs[row_idx, 0].set_title(f'{datum0["patient_slice_name"]}: percentage')
        axs[row_idx, 1].set_title(f'{datum0["patient_slice_name"]}: label before')
        axs[row_idx, 2].set_title(f'{datum0["patient_slice_name"]}: label after')
        
        if datum1_idx > len(data_with_scar) - 1:
            break
        datum1 = data_with_scar[datum1_idx]        
        axs[row_idx, 3].plot(datum1['scar_sector_percentage']*10, np.arange(len(datum0['scar_sector_label'])), linewidth=2)
        axs[row_idx, 4].plot(datum1['scar_sector_label']*10, np.arange(len(datum0['scar_sector_label'])), linewidth=2)
        axs[row_idx, 5].plot(datum1['scar_sector_label_connected']*10, np.arange(len(datum0['scar_sector_label'])), linewidth=2)
        
        
        axs[row_idx, 3].pcolor(np.squeeze(datum1['strainMatFullResolutionSVD'][...,:-5]), vmax = 0.2, vmin = -0.2, cmap='jet')
        axs[row_idx, 4].pcolor(np.squeeze(datum1['strainMatFullResolutionSVD'][...,:-5]), vmax = 0.2, vmin = -0.2, cmap='jet')
        axs[row_idx, 5].pcolor(np.squeeze(datum1['strainMatFullResolutionSVD'][...,:-5]), vmax = 0.2, vmin = -0.2, cmap='jet')
                
        
        axs[row_idx, 3].set_title(f'{datum1["patient_slice_name"]}: percentage')
        axs[row_idx, 4].set_title(f'{datum1["patient_slice_name"]}: label before')
        axs[row_idx, 5].set_title(f'{datum1["patient_slice_name"]}: label after')
        
        
        
        # axs[row_idx, 0].plot(data_with_scar[plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 0]['scar_sector_label'])
        # axs[row_idx, 1].plot(data_with_scar[plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 0]['scar_sector_label_connected'])
        # axs[row_idx, 2].plot(data_with_scar[plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 1]['scar_sector_label'])
        # axs[row_idx, 3].plot(data_with_scar[plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 1]['scar_sector_label_connected'])
        
        # axs[row_idx, 0].set_title(f'{plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 0}: before')
        # axs[row_idx, 1].set_title(f'{plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 0}: after')
        # axs[row_idx, 2].set_title(f'{plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 1}: before')
        # axs[row_idx, 3].set_title(f'{plot_idx*n_rows*n_cols//2 + row_idx*n_cols//2 + 1}: after')
        
        
# %% Quickly plot data