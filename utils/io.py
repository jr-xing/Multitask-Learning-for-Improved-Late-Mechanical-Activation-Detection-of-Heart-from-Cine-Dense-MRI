# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:12:32 2021

@author: Jerry Xing
"""
from pathlib import Path
import os

def create_folder(folder_path, recursive=False, action_when_exist='add index'):
    target_path = Path(folder_path)
    if target_path.is_dir() == False:
        # If path does not exist
        target_path.mkdir(parents=recursive)
        return folder_path# + '\\'
    else:
        # if path already exists
        if action_when_exist == 'pass':
            return folder_path
        elif action_when_exist == 'add index':
            for idx in range(2, 100):
                folder_path_idxed = folder_path + '-' + str(idx)
                if not Path(folder_path_idxed).is_dir():
                    Path(folder_path_idxed).mkdir()
                    break

            return folder_path_idxed #+ '\\'


# %% Load data according to excel file
import numpy as np
import scipy.io as sio
import pandas as pd
from utils.strainmat_utils import getStrainMatFull, SVDDenoise
from tqdm import tqdm

def loadmat(filename):
    return sio.loadmat(filename, struct_as_record=False, squeeze_me=True)


def extract_data_types_from_DENSEanalysis_file(mat, data_types, load_TOS=True):
    data = {}

    def warn_under_17(x):
        # x[x < 17] = 17
        if np.min(x) < 17:
            print('<17!')
        return x
    
    data_types += ['slice_spatial_location']

    for data_type in data_types:
        if data_type in ['strainmat']:
            data[data_type] = mat['TransmuralStrainInfo'].Ecc.mid[None, None, :, :]
        elif data_type in ['strainmatSVD']:
            data[data_type] = mat['TransmuralStrainInfo'].Ecc.midSVD[None, None, :, :]
        elif data_type in ['strainMatFullResolution']:
            try:
                data[data_type] = mat['StrainInfo'].CCmid[None, None, :, :]
            except:
                data[data_type] = getStrainMatFull(mat)[None, None, :, :]
        elif data_type in ['strainMatFullResolutionSVD']:
            try:
                data[data_type] = mat['StrainInfo'].CCmidSVD[None, None, :, :]
            except:
                data[data_type] = SVDDenoise(getStrainMatFull(mat))[None, None, :, :]
        elif data_type in ['TOS', 'TOS18_Jerry', 'TOSfullRes_Jerry', 'TOS126', 'TOS18']:
            if load_TOS:
                try:
                    if data_type in ['TOS']:
                        data[data_type] = mat['TOSAnalysis'].TOS[None, :]
                    elif data_type in ['TOS18_Jerry']:
                        data[data_type] = warn_under_17(mat['TOSAnalysis'].TOS18_Jerry[None, :])
                    elif data_type in ['TOSfullRes_Jerry']:
                        data[data_type] = warn_under_17(mat['TOSAnalysis'].TOSfullRes_Jerry[None, :])
                except:
                    print('Warning: no TOS found in DENSE MAT')
            else:
                continue
        # elif data_type in ['TOS18', 'TOS126']:
        #     data[data_type] = warn_under_17(mat['TOSAnalysis'].TOS18_Jerry[None, :])
        elif data_type.split('=')[0] in ['scar-AHA-step', 'scar-AHA-distmap']:
            continue
        elif data_type.split('=')[0] in ['scar_sector_percentage', 'scar_sector_label', 'scar_sector_distmap']:
            continue
        elif data_type in ['late_activation_sector_label']:
            continue
        elif data_type.split('=')[0] in ['has_scar']:
            continue
        elif data_type in ['slice_spatial_location']:
            data[data_type] = mat['SequenceInfo'][0,0].SliceLocation
        else:
            raise ValueError(f'Unsupported data type: {data_type}')
    return data


def get_full_filename_from_slice_record(slice_record, dataset_dir, data_type):
    # print(slice_record['Set Name'], str(slice_record['Patient Name']),
    #                 slice_record[data_type])
    return str(Path(dataset_dir, slice_record['Set Name'], str(slice_record['Patient Name']),
                    slice_record[data_type]))
    # print('AA')
    # print(os.path.join(dataset_dir, slice_record['Set Name'], str(slice_record['Patient Name']),
    #                 slice_record[data_type]))
    # return os.path.join(dataset_dir, slice_record['Set Name'], str(slice_record['Patient Name']),
    #                 slice_record[data_type])

def extract_DENSE_data_from_single_slice_record(slice_info: pd.Series, dataset_dir: str, data_types=None, load_TOS=True):
    patient_slice_mat_filename = get_full_filename_from_slice_record(slice_info, dataset_dir,
                                                                     'Strain Data Path under Patient Directory')
    # print(patient_slice_mat_filename)    
    try:
        mat = sio.loadmat(patient_slice_mat_filename, struct_as_record=False, squeeze_me=True)
    except:
        print('failed to load ', patient_slice_mat_filename)
        return {}
    slice_data = extract_data_types_from_DENSEanalysis_file(mat, data_types, load_TOS)
    slice_data['DENSE_filename'] = patient_slice_mat_filename
    # slice_data['patient_name'] = slice_info['Set Name'] + '-' + str(slice_info['Patient Name'])
    # slice_data['slice_name'] = slice_info['Slice Name']
    # slice_data['patient_slice_name'] = slice_data['patient_name'] + '-' + slice_data['slice_name']
    # slice_data['hasScar'] = slice_info['Scar']
    # slice_data['spatial_location'] = mat['SequenceInfo'][0, 0].SliceLocation
    return slice_data

def extract_DENSE_data_from_slices_records(slices_records, dataset_dir: str, data_types=None, load_TOS=True):
    DENSE_data = []
    if type(slices_records) is pd.DataFrame:
        for slice_info_idx, slice_info in slices_records.iterrows():
            # patient_slice_mat_filename = str(Path(dataset_dir, patient_slice_info['Set Name'], patient_slice_info['Patient Name'], patient_slice_info['Strain Data Path under Patient Directory'].replace('_Jerry', '')))
            slice_data = extract_DENSE_data_from_single_slice_record(slice_info, dataset_dir, data_types, load_TOS)            
            DENSE_data.append(slice_data)
    else:
        slice_data = extract_DENSE_data_from_single_slice_record(slices_records, dataset_dir, data_types, load_TOS)
        DENSE_data.append(slice_data)
    return DENSE_data


def extract_scar_enhancement_data_from_patient_AHA_records(patient_scar_annotation_AHA_records: pd.DataFrame,
                                                           dataset_dir: str):
    AHA_sector_name_order_6 = ['Inferoseptum', 'Inferior', 'Inferolateral', 'Anterolateral', 'Anterior', 'Anteroseptum']
    AHA_sector_name_order_4 = ['Septum', 'Inferior', 'Lateral', 'Anterior']
    patient_slice_sector_names = list(patient_scar_annotation_AHA_records['Label'])
    patient_slice_names = np.unique(
        [slice_sector_name.split(':')[0] for slice_sector_name in patient_slice_sector_names])
    patient_enhancement_info = []

    # Extract sector data for each slice
    for patient_slice_name in patient_slice_names:
        # Get rows of current slice
        patient_slice_rows = patient_scar_annotation_AHA_records[
            patient_scar_annotation_AHA_records['Label'].str.startswith(patient_slice_name + ':')]
        if len(patient_slice_rows) == 6:
            AHA_sector_name_order = AHA_sector_name_order_6
            slice_enhancement = np.zeros(6)
        elif len(patient_slice_rows) == 4:
            AHA_sector_name_order = AHA_sector_name_order_4
            slice_enhancement = np.zeros(4)
        else:
            raise ValueError(
                f'Unrecognized number of sector of slice {patient_slice_name}, except 4 ro 6, got {len(patient_slice_rows)}')
        # Load data of each sector of current slice
        for AHA_sector_idx, AHA_sector_name in enumerate(AHA_sector_name_order):
            slice_enhancement[AHA_sector_idx] = \
                patient_slice_rows[patient_slice_rows['Label'].str.contains(AHA_sector_name)].iloc[0]['Enhancement']
        # Append data
        patient_enhancement_info.append({
            'slice_name': patient_slice_name,
            'slice_enhancement': slice_enhancement,
            'slice_spatial_location': patient_slice_rows.iloc[0]['Spatial Location (Jerry)']
        })
    # print(patient_enhancement_info)
    return patient_enhancement_info


def attach_scar_annotation_to_DENSE_slices(patient_DENSE_data: list, patient_scar_annotation: list,
                                           patient_scar_slices_spatial_location_shift=0):
    patient_scar_annotation_slice_spatial_locations = \
        np.array([patient_slice_scar_annotation['slice_spatial_location'] for patient_slice_scar_annotation in
                  patient_scar_annotation]) - patient_scar_slices_spatial_location_shift

    for patient_slice_DENSE_data in patient_DENSE_data:
        # print(patient_slice_DENSE_data.keys())
        patient_slice_spatial_locaiton = patient_slice_DENSE_data['spatial_location']
        patient_slice_matching_scar_annotation = patient_scar_annotation[np.argmin(
            np.abs(patient_slice_spatial_locaiton - patient_scar_annotation_slice_spatial_locations))]
        patient_slice_DENSE_data['scar_AHA_raw'] = patient_slice_matching_scar_annotation['slice_enhancement']
        patient_slice_DENSE_data['scar_AHA_raw_spatial_location'] = patient_slice_matching_scar_annotation[
            'slice_spatial_location']
    # return patient_DENSE_data

def load_TOS_from_file(filename, keep_original_key=True, add_additional_dim=False):
    # print(PurePath(filename))
    # TOS_mat = sio.loadmat(filename.replace("\/", "a"))
    try:
        TOS_mat = sio.loadmat(filename)
    except:
        print('Error loading ', filename)
        return {}
    
    def ensure_minimum_equals_17(TOS):
        TOS[TOS < 17] = 17
        return TOS

    # Initialize data with None
    # TOS_data = {'18': None, '126': None}
    TOS_data = {}
    # Load TOS 18
    TOS_18_keys_in_order = ['TOS18', 'xs18_new', 'xs']
    for TOS_18_key in TOS_18_keys_in_order:
        if TOS_18_key in TOS_mat.keys():
            TOS_18 = ensure_minimum_equals_17(TOS_mat[TOS_18_key])
            if add_additional_dim:
                TOS_18 = TOS_18[None,:]
            
            if keep_original_key:
                TOS_data[TOS_18_key] = TOS_18
            else:
                TOS_data['18'] = TOS_18

            break

    # Load TOS 126
    TOS_126_keys_in_order = ['TOS126', 'xsfullRes_new', 'xsfullRes']
    for TOS_126_key in TOS_126_keys_in_order:
        if TOS_126_key in TOS_mat.keys():
            if keep_original_key:
                TOS_data[TOS_126_key] = ensure_minimum_equals_17(TOS_mat[TOS_126_key])
            else:
                TOS_data['126'] = ensure_minimum_equals_17(TOS_mat[TOS_126_key])
            break
    return TOS_data

def load_scar_from_file(filename):
    scar_df = pd.read_csv(filename)
    scar_percentage = scar_df['scar_percentage'].to_numpy()
    return scar_percentage

# def fiulter_data_by_patient():
#     pass

def load_data_from_table(data_records_filename=None, dataset_dir=None, pool_size=8, load_num=None, data_info=None,\
                         load_DENSE=True, load_TOS=True, load_scar=True, load_unused=False,
                         include_patient_names=None, exclude_patient_names=None):
    # https://reijz.github.io/blog/python-read-data-parallel/
    # Set default data record filename
    if dataset_dir is None:
        dataset_dir = 'D:\\Documents\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry'
        # dataset_dir = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry'
    
    if data_records_filename is None:
        data_records_filename = str(Path(dataset_dir, 'record_sheets\\cardiac-strainmat-dataset-2021-05-31.xlsx'))
        
    # print(data_records_filename)
    # Load data record file
    if data_records_filename.endswith('.xlsx'):
        data_records = pd.read_excel(data_records_filename, engine='openpyxl')
    else:
        data_records = pd.read_csv(data_records_filename)
    
    if not load_unused:
        data_records_to_use = data_records[data_records['Use'] == 1]
    else:
        data_records_to_use = data_records
    
    def df_slicing(df: pd.DataFrame, slicing_indices: int or list):
        if type(slicing_indices) is int:
            if slicing_indices > 0:
                df_sliced = df.iloc[:slicing_indices]
            elif slicing_indices < 0:
                df_sliced = df.iloc[slicing_indices:]
        elif type(slicing_indices) in [list, tuple]:
            df_sliced = pd.concat([df_slicing(df, slicing_idx) for slicing_idx in slicing_indices])
            
        return df_sliced
    
    if load_num is not None:
        data_records_to_use = df_slicing(data_records_to_use, load_num)
        
    data = []
    # For each slice record
    for slice_idx, slice_record in data_records_to_use.iterrows():
    # for slice_idx, slice_record in data_records_to_use.iloc[-20:].iterrows():
    # for slice_idx, slice_record in tqdm(data_records_to_use.iterrows(), total=data_records_to_use.shape[0]):
        slice_data = {}
        slice_data['set_name'] = slice_record['Set Name']
        slice_data['patient_name'] = slice_record['Set Name'] + '-' + str(slice_record['Patient Name'])
        slice_data['slice_name'] = slice_record['Slice Name']
        slice_data['patient_slice_name'] = slice_data['patient_name'] + '-' + slice_data['slice_name']
        slice_data['hasScar'] = slice_record['Scar']
        
        
        if include_patient_names is not None and slice_data['patient_name'] not in include_patient_names:
            # print('Skip: ', slice_data['patient_name'])
            continue
        
        if exclude_patient_names is not None and slice_data['patient_name'] in exclude_patient_names:
            continue        
        
        # if True:
        #     print(True)
        
        # Load DENSE data
        if data_info is not None:
            # if type(data_info) is list:
            if type(data_info[0]) is str:
                data_types = data_info
            else:
                data_types = [data_type['type'] for data_type in data_info]
        else:
            data_types = None
        
        load_TOS_from_strainmat_file = slice_record['Use External TOS File'] != 1        
        if load_DENSE:
            slice_DENSE_data = extract_DENSE_data_from_slices_records(slice_record, dataset_dir, data_types, load_TOS=load_TOS_from_strainmat_file)[0]
            slice_data.update(slice_DENSE_data)

        # Load TOS file if exists
        if not pd.isna(slice_record['TOS Data Path under Patient Directory']) and load_TOS:
            TOS_filename = get_full_filename_from_slice_record(slice_record, dataset_dir, 'TOS Data Path under Patient Directory')
            slice_data['TOS_filename'] = TOS_filename
            TOS_data = load_TOS_from_file(TOS_filename, keep_original_key=True)
            slice_data.update(TOS_data)

        # Load scar data if exists
        if slice_record['Scar'] == 1 and load_scar:
            if slice_record['Use-Scar-AHA-Annotation'] == 1 and not pd.isna(slice_record['Scar-Annotation-AHA-filename']):
                pass
            elif not pd.isna(slice_record['Scar-126-filename']):
                print('loading scar', slice_record['Patient Name'], slice_record['Slice Name'])
                scar_filename = get_full_filename_from_slice_record(slice_record, dataset_dir, 'Scar-126-filename')
                scar_sector_percentage = load_scar_from_file(scar_filename)
                slice_data['scar_sector_percentage'] = scar_sector_percentage

        data.append(slice_data)
    return data

def load_data_by_config(config, dataset_dir=None, data_records_filename=None, load_training_data=False, load_test_data=True, generate_dataset=False):
    from utils.io import load_data_from_table
    from pathlib import PurePath
    if dataset_dir is None:
        dataset_dir = str(PurePath('D:\\Documents\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac', 'Dataset/CRT_TOS_Data_Jerry'))
    # load_data_num = [20, -20]
    load_data_num = None
    included_data_info = config['data']['input_info'] + config['data']['output_info']
    included_data_types = [data_type['type'] for data_type in included_data_info]
    # dataFilename = config['data']['filename']
    if data_records_filename is None:
        data_records_filename = str(PurePath('D:\\Documents\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac', config['data']['filename'].replace('../../', '')))
    
    if load_training_data == False and load_test_data == True:
        load_patient_names = config['data']['train_test_split']['paras']['test_patient_names']
    else:
        load_patient_names = None
    dataFull = load_data_from_table(data_records_filename, 
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
        if generate_dataset:
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
        else:
            training_dataset = None
            test_dataset = None
    else:
        from modules.dataset import Dataset
        import numpy as np
        dataset_precision = np.float16
        
        training_data_raw = []
        training_data_aug = []
        training_data = []
        test_data = dataFull
        if generate_dataset:
            training_dataset = None            
            test_dataset = Dataset(test_data, config['data']['input_info'], config['data']['output_info'], precision=dataset_precision)
            test_dataset.input_info = config['data']['input_info']
            test_dataset.output_info = config['data']['output_info']
        else:
            training_dataset = None
            test_dataset = None
    
    return {
        'training_data': training_data,
        'training_data_raw': training_data_raw,
        'training_data_aug': training_data_aug,
        'test_data': test_data,
        'training_dataset': training_dataset,
        'test_dataset': test_dataset
        }
    # if return_dataset:
    #     return training_dataset, test_dataset
    # else:
    #     return training_data, test_data

def load_data_from_table_old(data_records_filename=None, dataset_dir=None, pool_size=8, load_num=-1, data_info=None):
    # https://reijz.github.io/blog/python-read-data-parallel/
    # Set default data record filename
    if dataset_dir is None:
        dataset_dir = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry'
    if data_records_filename is None:
        data_records_filename = str(Path(dataset_dir, 'record_sheets\\cardiac-strainmat-dataset-2021-05-31.xlsx'))

    # Load data record file
    if data_records_filename.endswith('.xlsx'):
        data_records = pd.read_excel(data_records_filename, engine='openpyxl')
    else:
        data_records = pd.read_csv(data_records_filename)
    data_records_to_use = data_records[data_records['Use'] == 1]
    if load_num > 0:
        data_records_to_use = data_records_to_use.iloc[:load_num]
    # data_records_with_scar_AHA = data_records[data_records['Scar-Annotation-AHA-filename'].notnull()]
    # Load data by patient, since not all data are provided slice by slice, or provided using different slicing locations
    unique_patient_records = data_records_to_use[['Set Name', 'Patient Name']].drop_duplicates()

    data = []
    # For data of each patients
    # Organize by patient because the AHA annotation is organized by patient
    for patient_idx, patient_records in unique_patient_records.iterrows():
        # Get records of current patient
        patient_set = patient_records['Set Name']
        patient_name = patient_records['Patient Name']
        patient_slices_records = data_records_to_use[data_records_to_use['Patient Name'] == patient_name]

        # Load DENSE data
        if data_info is not None:
            data_types = [data_type['type'] for data_type in data_info]
        else:
            data_types = None
        patient_DENSE_data = extract_DENSE_data_from_slices_records(patient_slices_records, dataset_dir,
                                                                    data_types)

        # Load scar annotation data if exists
        if patient_slices_records.iloc[0]['Use-Scar-AHA-Annotation'] == 1:
            # Load AHA model label
            patient_scar_annotation_AHA_filename = patient_slices_records.iloc[0]['Scar-Annotation-AHA-filename']
            if not pd.isnull(patient_scar_annotation_AHA_filename):
                # If file exists
                patient_scar_annotation_AHA_filename_full = \
                    str(Path(dataset_dir, patient_set, patient_name, patient_scar_annotation_AHA_filename))
                if patient_scar_annotation_AHA_filename.endswith('.xlsx'):
                    patient_scar_annotation_AHA_records = pd.read_excel(
                        patient_scar_annotation_AHA_filename_full,
                        engine='openpyxl')
                else:
                    patient_scar_annotation_AHA_records = pd.read_csv(
                        patient_scar_annotation_AHA_filename_full)
                patient_scar_annotation_AHA = \
                    extract_scar_enhancement_data_from_patient_AHA_records(patient_scar_annotation_AHA_records,
                                                                           dataset_dir)
            else:
                # If file doesn't exist
                patient_scar_annotation_AHA = None
        elif patient_slices_records.iloc[0]['Use-Scar-AHA-Annotation'] == 0:
            # Load from file (126 table)
            pass
        elif patient_slices_records.iloc[0]['Use-Scar-AHA-Annotation'] == -1:
            # No scar
            pass
        else:
            raise ValueError(
                'Unsupported scar source type: ' + str(patient_slices_records.iloc[0]['Use-Scar-AHA-Annotation']))

        # patient_scar_annotation_AHA_filename = patient_slices_records.iloc[0]['Scar-Annotation-AHA-filename']
        # if not pd.isnull(patient_scar_annotation_AHA_filename):
        #     patient_scar_annotation_AHA_filename_full = \
        #         str(Path(dataset_dir, patient_set, patient_name, patient_scar_annotation_AHA_filename))
        #     if patient_scar_annotation_AHA_filename.endswith('.xlsx'):
        #         patient_scar_annotation_AHA_records = pd.read_excel(
        #             patient_scar_annotation_AHA_filename_full,
        #             engine='openpyxl')
        #     else:
        #         patient_scar_annotation_AHA_records = pd.read_csv(
        #             patient_scar_annotation_AHA_filename_full)
        #     patient_scar_annotation_AHA = \
        #         extract_scar_enhancement_data_from_patient_AHA_records(patient_scar_annotation_AHA_records, dataset_dir)
        # else:
        #     patient_scar_annotation_AHA = None

        # For each DENSE slice, try finding the scar annotation with matching spatial location
        if patient_scar_annotation_AHA is not None:
            attach_scar_annotation_to_DENSE_slices(patient_DENSE_data, patient_scar_annotation_AHA)
        else:
            for patient_slice_DENSE_data in patient_DENSE_data:
                patient_slice_DENSE_data['scar_AHA_raw'] = None
                patient_slice_DENSE_data['scar_AHA_raw_spatial_location'] = None
        data += patient_DENSE_data
    return data
