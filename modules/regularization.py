# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:28:20 2021

@author: remus
"""

# for key in regularization.keys():
#                         if key.lower() == 'l1':
#                             # https://stackoverflow.com/questions/46797955/l1-norm-as-regularizer-in-pytorch
#                             # https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
#                             L1_weight = regularization[key]
#                             L1_reg = torch.tensor(0., requires_grad=True).to(self.device, dtype = torch.float)
#                             for name, param in self.net.named_parameters():
#                                 if 'weight' in name:
#                                     L1_reg = L1_reg + torch.norm(param, 1)
                            
#                             loss += L1_weight * L1_reg
import torch
def regularize(network, method:str, paras:dict = {}, device = torch.device("cpu")):
    if method is None:
        return 0
    elif method.lower() == 'l2':
        # https://stackoverflow.com/questions/46797955/l1-norm-as-regularizer-in-pytorch
        # https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
        weight = paras['weight']
        reg = torch.tensor(0., requires_grad=True).to(device, dtype = torch.float)
        for name, param in network.named_parameters():
            if 'weight' in name:
                reg = reg + torch.norm(param, 2)
        
        return weight * reg
    elif method.lower() == 'l1':
        # https://stackoverflow.com/questions/46797955/l1-norm-as-regularizer-in-pytorch
        # https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model
        L1_weight = paras['weight']
        L1_reg = torch.tensor(0., requires_grad=True).to(device, dtype = torch.float)
        for name, param in network.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)
        
        # loss += L1_weight * L1_reg
        return L1_weight * L1_reg