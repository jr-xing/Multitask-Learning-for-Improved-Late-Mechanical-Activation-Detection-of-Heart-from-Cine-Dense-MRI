# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:04:18 2021

Functions related to cardiac AHA model

@author: Jerry Xing
"""
# ----------------------------------------------------------------#
# ----------------------- FUNCTIONS ------------------------------#
# ----------------------------------------------------------------#
import pandas as pd
from scipy import interpolate
def AHA_enhancement_interpolation(slice_enhancement_array, target_dim=126, method='nearest', shift_4_segment=True):
    # Interpolate AHA enhancement data to array of target_dim
    # slice_enhancement: numpy ndarry of size (4,) or (6,)
    # shift_4_segment: if shift_4_segment == True, will shift the interpolation result since the the beginning sector is at the middle of first AHA segment
    
    # Triplicate to keep border continuous
    slice_enhancement_triplicated = np.tile(slice_enhancement_array, 3)
    dim_each_sector = target_dim / len(slice_enhancement_array)
    slice_enhancement_triplicated_locs = [dim_each_sector * (0.5 + idx) for idx in
                                          range(len(slice_enhancement_triplicated))]
    
    # Interpolate
    f = interpolate.interp1d(slice_enhancement_triplicated_locs, slice_enhancement_triplicated,
                             fill_value='extrapolate', kind=method)
    slice_enhancement_triplicated_interp_locs = np.arange(target_dim * 3)
    slice_enhancement_triplicated_interp = f(slice_enhancement_triplicated_interp_locs)
    slice_enhancement_interp = slice_enhancement_triplicated_interp[target_dim:target_dim * 2]
    
    # Shift 4-segment model interpolation result
    if len(slice_enhancement_array) == 4 and shift_4_segment:
        slice_enhancement_interp = np.roll(slice_enhancement_interp, target_dim // (4*2))
    
    # Filter out "wrong" small values to interploation
    slice_enhancement_interp[slice_enhancement_interp < np.min(slice_enhancement_array)] = np.min(slice_enhancement_array)
        
    return slice_enhancement_interp

def AHA_enhancement_to_sector_label(slice_enhancement: pd.DataFrame or dict, target_dim=126, threshold=-1):
    # The input slice_enhancement should contain the enhancement data of ALL segments (4 or 6) of a slice
    # It conould be a pandas DataFrame where each row has the segment name
    # or a dict of which the keys are segment names and the values are enhancement scalar
    n_segments = len(slice_enhancement)
    slice_enhancement_array = np.zeros(n_segments)
    segment_6_name_in_order = ['Inferoseptum', 'Inferior', 'Inferolateral', 'Anterolateral', 'Anterior', 'Anteroseptum']
    segment_4_name_in_order = ['Septum', 'Inferior', 'Lateral', 'Anterior']
    
    def get_segment_data_from_slice_dataFrame(segment_name: str, slice_DataFrame: pd.DataFrame):
        segment_row = slice_DataFrame[slice_DataFrame['Label'].str.contains(segment_name)]
        if len(segment_row) == 0:
            raise ValueError('segment not found!')
        elif len(segment_row) > 1:
            raise ValueError('multiple segment found!')
        else:
            return segment_row['Enhancement']
    
    def get_segment_data_from_slice_dict(segment_name: str, slice_dict: dict):
        segment_keys = [key for key in slice_dict.keys() if segment_name in key]
        if len(segment_keys) == 0:
            raise ValueError('segment not found!')
        elif len(segment_keys) > 1:
            raise ValueError('multiple segment found!')
        else:
            return slice_dict[segment_keys[0]]
    
    def get_segment_data_from_slice_data(segment_name: str, slice_data: pd.DataFrame or dict):
        if type(slice_data) is pd.DataFrame:
            return get_segment_data_from_slice_dataFrame(segment_name, slice_data)
        elif type(slice_data) is dict:
            return get_segment_data_from_slice_dict(segment_name, slice_data)
        else:
            raise ValueError(f'Unsupported data type: {type(slice_data)}')
    
    segment_names = segment_6_name_in_order if n_segments == 6 else segment_4_name_in_order
    for segment_idx, segment_name in enumerate(segment_names):
        slice_enhancement_array[segment_idx] = get_segment_data_from_slice_data(segment_name, slice_enhancement)
    
    slice_enhancement_array_interp = AHA_enhancement_interpolation(slice_enhancement_array, target_dim = target_dim)
    slice_enhancement_array_interp_binary = np.zeros_like(slice_enhancement_array_interp)
    slice_enhancement_array_interp_binary[slice_enhancement_array_interp >= threshold] = 1
    
    AHA_data = {
        'enhancement_raw': slice_enhancement_array,
        'enhancement_interpolated': slice_enhancement_array_interp,
        'enhancement_interpolated_binary': slice_enhancement_array_interp_binary
        }
    return AHA_data
# %%
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

# ----------------------------------------------------------------#
# ------------------------- TESTS --------------------------------#
# ----------------------------------------------------------------#
# %%
if __name__ == '__main__':
    # %% Load AHA scar annotation data
    import pandas as pd
    import numpy as np
    # AHA_file = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT16\\Scar-Annotation-AHA\\CT16-Late Enhancement - Polar Plot Summary Table.csv'
    AHA_file = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET03\\UP34\\Scar-Annotation-AHA\\UP34-Late Enhancement - Polar Plot Summary Table.csv'
    AHA_table = pd.read_csv(AHA_file)
    AHA_unique_slice_names = list(np.unique([row['Label'].split(':')[0] for idx, row in AHA_table.iterrows()]))
         # 1		patient_name	slice_name	scar_area	myocardium_area	scar_percentage
    
    # scar_dict = {
    #     'patient_name': '',
    #     'sclie_name': '',
    #     'scar_area': np.zeros(126),
    #     'myocardium_area': np.zeros(126),
    #     'scar_percentage': np.zeros(126)}
    AHA_data = []
    for AHA_unique_slice_name in AHA_unique_slice_names:
        AHA_slice_rows = AHA_table[AHA_table['Label'].str.startswith(AHA_unique_slice_name)]    
        AHA_slice_enhancement = {}
        AHA_slice_enhancement['slice_name'] = AHA_slice_rows.iloc[0]['Label'].split(':')[0]
        AHA_slice_enhancement['spatial_location'] = AHA_slice_rows.iloc[0]['Spatial Location (Jerry)']
        AHA_slice_enhancement.update(AHA_enhancement_to_sector_label(AHA_slice_rows))
        AHA_data.append(AHA_slice_enhancement)
    # %% Load DENSE Data
    import os
    import scipy.io as sio
    strainmat_folder = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET03\\UP34\\mat\\'
    DENSE_data = []
    for strainmat_filename in os.listdir(strainmat_folder):
        strainmat_mat = sio.loadmat(strainmat_folder + strainmat_filename, struct_as_record=False, squeeze_me=True)
        # strainmat_mat = sio.loadmat(strainmat_folder + strainmat_filename, struct_as_record=True, squeeze_me=True)
        # spl = strainmat_mat['SequenceInfo'][0,0]['SliceLocation'][0,0]
        DENSE_datum = {
            'slice_name': strainmat_filename.split('.mat')[0],
            'spatial_location':strainmat_mat['SequenceInfo'][0,0].SliceLocation,
            'strain_matrix': strainmat_mat['StrainInfo'].CCmid
            }
        DENSE_data.append(DENSE_datum)
    
    # %% Match DENSE data and AHA scar annotation
    DENSE_locations = np.array([DENSE_datum['spatial_location'] for DENSE_datum in DENSE_data])
    AHA_locations = np.array([AHA_datum['spatial_location'] for AHA_datum in AHA_data])
    print('DENSE_locations: ', DENSE_locations)
    print('AHA_locations: ', AHA_locations)
    
    for DENSE_datum_idx, DENSE_datum in enumerate(DENSE_data):
        # For each DENSE slice, find the AHA slice with closest spatial location
        closest_AHA_slice_idx = np.argmin(np.abs(DENSE_datum['spatial_location'] - AHA_locations))
        closest_AHA_slice = AHA_data[closest_AHA_slice_idx]
        DENSE_datum['AHA'] = closest_AHA_slice
        print(f'DENSE {DENSE_datum_idx} -> AHA {closest_AHA_slice_idx}')
        
    # %% Show results
    import matplotlib.pyplot as plt
    n_cols = 4
    n_rows = int(np.ceil(len(DENSE_data) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols)
    axs = axs.flatten()
    for DENSE_datum_idx, DENSE_datum in enumerate(DENSE_data):
        axe = axs[DENSE_datum_idx]
        axe.pcolor(DENSE_datum['strain_matrix'], vmax=0.2, vmin=-0.2, cmap='jet')
        axe.plot(DENSE_datum['AHA']['enhancement_interpolated'] / 10, np.arange(126))
        print(DENSE_datum['AHA']['enhancement_raw'])
        