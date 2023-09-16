# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:20:13 2021

@author: Jerry Xing
"""
#%%
import numpy as np
dataFilename = 'D://dataFull-201-2020-12-14-Jerry.npy'
dataSaved = np.load(dataFilename, allow_pickle = True).item()
dataInfo = dataSaved['description']
dataFull = dataSaved['data']
scarfree = [datum for datum in dataFull if datum['hasScar'] == False and 'SETOLD' not in datum['dataFilename']]
scarwith = [datum for datum in dataFull if datum['hasScar'] == True]

#%%
from utils.data import getPatientName
scar_free_patient_names = np.unique([getPatientName(datum['dataFilename']) for datum in scarfree if 'SETOLD' not in datum['dataFilename']])
scar_with_patient_names = np.unique([getPatientName(datum['dataFilename']) for datum in scarwith])
