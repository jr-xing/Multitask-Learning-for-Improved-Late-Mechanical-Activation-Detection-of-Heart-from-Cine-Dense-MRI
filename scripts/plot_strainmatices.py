# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 15:01:33 2021

@author: Jerry Xing
"""

# Plotting strain matrix and TOS to check the correctness of TOS

# %%1. Load data
from utils.io import load_data_from_table
# data_records_filename = str(Path("D:\Documents\OneDrive\Documents\Study\Researches\Projects\cardiac\Dataset\CRT_TOS_Data_Jerry\cardiac-strainmat-dataset-2021-06-13-scar-classification.xlsx"))
# data_records_filename = data_records_filename.replace('CRT_TOS_Data_Jerry\record_sheets', 'CRT_TOS_Data_Jerry\\record_sheets')
data_records_filename = "D:\\Documents\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\record_sheets\\cardiac-strainmat-dataset-2021-06-13-scar-classification.xlsx"
data_info = [{'type':'strainMatFullResolutionSVD'}, {'type': 'TOS126'}]
data = load_data_from_table(data_records_filename = data_records_filename, 
                            load_all = True,
                            data_info = data_info)

# %% Filter patient
# data_to_show = data
include_set_names = []
exclude_set_names = ['Pre_CRT_LBBB_with_scar']
include_set_patient_names = []
exclude_set_patient_names = []
include_patient_slice_names = []
exclude_patient_slice_names = []

def check_if_include_data(
        datum, include_set_names=[],
        exclude_set_names=[],
        include_set_patient_names=[],
        exclude_set_patient_names=[],
        include_patient_slice_names=[],
        exclude_patient_slice_names=[],
        other_conditions=[]
        ):
    if len(include_set_names) > 0 and datum['set_name'] not in include_set_names:
        return False
    
    if len(exclude_set_names) > 0 and datum['set_name'] in exclude_set_names:
        return False
    
    if len(include_set_patient_names) > 0 and datum['patient_name'] not in include_set_patient_names:
        return False
    
    if len(include_set_patient_names) > 0 and datum['patient_name'] in exclude_set_patient_names:
        return False
    
    if len(include_patient_slice_names) > 0 and datum['set_name'] not in include_patient_slice_names:
        return False
    
    if len(exclude_patient_slice_names) > 0 and datum['set_name'] in exclude_patient_slice_names:
        return False
    
    for other_contidion in other_conditions:
        if other_contidion == 'should_have_scar' and datum.get('hasScar', False):
            return False
    
    return True

data_to_show = [datum for datum in data if check_if_include_data(datum, include_set_names, exclude_set_names, include_set_patient_names, exclude_set_patient_names, include_patient_slice_names, exclude_patient_slice_names)]
data_to_show = [datum for datum in data_to_show if 'TOS126' in datum.keys() and 'strainMatFullResolutionSVD' in datum.keys()]

# %% 
from pathlib import Path
from utils.io import create_folder
plot_dir = "D:\\Research\\Cardiac\\strainmat_images\\July-03-2021\\"
create_folder(plot_dir, recursive=False, action_when_exist='pass')

# %% 
import numpy as np
from utils.plot import save_multi_strainmat_with_curves
organize_method = 'by patient'
if organize_method == 'by patient':
    unique_patient_names = np.unique([datum['patient_name'] for datum in data_to_show])
    for patient_name in unique_patient_names:
        data_of_patient = [datum for datum in data_to_show if datum['patient_name'] == patient_name]
        save_multi_strainmat_with_curves(
            data = data_of_patient, 
            strainmat_type = 'strainMatFullResolutionSVD', 
            curve_types = ['TOS126'],
            save_filename = str(Path(plot_dir, patient_name + '.png')), \
            fig=None, axs=None, \
            legends=None, title=None, subtitles=[datum['patient_slice_name'] for datum in data_of_patient], \
            vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
            n_cols=4,
            enable_multipages=True, 
            n_rows_per_page=4, 
            colors=None)
