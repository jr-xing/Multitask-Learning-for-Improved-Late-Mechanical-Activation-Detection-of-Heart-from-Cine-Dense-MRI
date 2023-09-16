# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:00:35 2021

@author: Jerry Xing
"""
from modules.networks.strain_networks import *
# from modules.networks.strain_networks_old_02 import *
from modules.networks.polyfit import *
def get_network_by_name(name, config = {}):
    # if name =='NetStrainMatSectionClassify':        
    #     net = NetStrainMatSectionClassify(config)
    # elif name =='NetStrainMatJointClsPred':
    #     net = NetStrainMatJointClsPred(config)
    # elif name =='NetStrainMat2TOS':
    #     net = NetStrainMat2TOS(config)
    if name =='NetStrainMat2TOS':
        net = NetStrainMat2TOS(config)
    elif name == 'NetStrainMat2ClsReg':
        net =  NetStrainMat2ClsReg(config)
    elif name == 'NetStrainMat2ClsDistMapReg':
        net =  NetStrainMat2ClsDistMapReg(config)
    elif name == 'NetStrainMat2PolycoeffCls':
        net = NetStrainMat2PolycoeffCls(config)
    elif name == 'NetStrainMat2Cls':
        net = NetStrainMat2Cls(config)
    elif name == 'NetStrainMat2Reg':
        net = NetStrainMat2Reg(config)
    else:
        raise ValueError('Unsupported net name: ', name)
    return net