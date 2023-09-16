# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:17:23 2021

@author: remus
"""
# import random
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset


class DataSet2D(TorchDataset):
    """
    Load in 3D medical image, treate image as a stack of 2D images with given dimension
    """

    def __init__(self, imgs, labels, labelmasks=None, transform=None, device=torch.device("cpu")):
        super(DataSet2D, self).__init__()
        # img should be [N,H,W,C] or [N, T, H, W, C]
        data = imgs

        # Note that in transform pytorch assume the image is PILImage [H, W, C]!
        self.data = data.astype(np.float32)
        self.labels = labels
        if labelmasks is None:
            # labelmasks = ['None'] * len(data)
            # labelmasks = np.ones(len(data)) * np.nan
            # labelmasks = np.ones(len(data))
            labelmasks = np.ones(labels.shape)
        self.labelmasks = labelmasks
        self.transform = transform
        #        if transform != None:
        #            self.data = self.transform(self.data)
        #        self.data.to(dtype=torch.float)
        self.dataShape = self.data.shape

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        label = self.labels[idx]
        labelmask = self.labelmasks[idx]
        if self.transform is not None:
            sample = self.transform(sample)
            # return sample
        return {'data': sample, 'label': label, 'labelmask': labelmask}


class Dataset(TorchDataset):
    def __init__(self, data: list, input_info=None, output_info=None, precision=np.float32):
        # , device = torch.device("cpu")
        super(Dataset, self).__init__()
        # if input_types is None or output_types is None:
        #     # self.data = data
        #     self.data = [{data_type: datum[data_type] for data_type in input_types + output_types} for datum in data]
        #     input_types = None
        #     output_types = None
        # else:
        input_types = [info['type'] for info in input_info]
        output_types = [info['type'] for info in output_info]
        self.data = [{data_type: datum[data_type][0, :] for data_type in input_types + output_types} for datum in data]
        
        # for other_type in ['augmented', 'patient_name', 'slice_name']:
        #     for datum_idx in range(len(data)):
        #         self.data[datum_idx][other_type] = data[datum_idx][other_type]
            
        self.N = len(data)
        self.input_types = input_types
        self.output_types = output_types

        # Remove the N dimension to make sure the dimension is correct when getting batch data
        # (Depreciated since using mixed precision)(would it work?)
        for key in self.data[0].keys():
            if type(self.data[0][key]) is np.ndarray:
                for datum in self.data:
                    # datum[key] = datum[key][0,:].astype(precision)
                    datum[key] = datum[key].astype(precision)

        # self.device = device
        # if 'strainMat'

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return self.data[idx]
        # return {key: self.data[idx][key][0, :] for key in self.data[idx].keys()}
        return self.data[idx]


class DataSet1I1O(TorchDataset):
    def __init__(self, data: list, input_type: str, output_type: str, precision=np.float32):
        super(DataSet1I1O, self).__init__()
        # , device = torch.device("cpu")
        # print("!!")

        if type(input_type) is str:
            self.input_type = input_type
            self.input_types = [input_type]
        else:
            self.input_type = input_type[0]
            self.input_types = input_type

        if type(output_type) is str:
            self.output_type = output_type
            self.output_types = [output_type]
        else:
            self.output_type = output_type[0]
            self.output_types = output_type
        # if type(input_type) is list:
        #     input_type = input_type[0]
        # if type(output_type) is list:
        #     output_type = input_type[0]

        self.data = {}
        self.data[self.input_type] = np.concatenate([datum[self.input_type] for datum in data], axis=0).astype(
            precision)
        self.data[self.output_type] = np.concatenate([datum[self.output_type] for datum in data], axis=0).astype(
            precision)

        self.N = len(data)
        # self.input_type = input_type
        # self.output_type = output_type

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return self.data[idx]
        # return {key:self.data[idx][key][0,:] for key in self.data[idx].keys()}
        return {self.input_type: self.data[self.input_type][idx], self.output_type: self.data[self.output_type][idx]}
