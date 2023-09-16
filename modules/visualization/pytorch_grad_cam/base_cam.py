# import cv2
import numpy as np
import torch
import ttach as tta
from modules.visualization.pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from modules.visualization.pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from skimage.transform import resize as skresize
from icecream import ic
from modules.evaluation import evaluate
from utils.data import get_data_type_by_category
import copy

class BaseCAM:
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)

    # def forward(self, input_img):
    #     return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    # def get_loss(self, output, target_category, method = 'Jerry'):
    # def get_loss(self, output, target_category, sector_idx=0, method = 'cls'):            
    def get_output_sum(self, output, target_category, sector_idx=0, method = 'cls'):            
        loss = 0
        # print(output.shape)
        if method == 'ori':
            for i in range(len(target_category)):
                loss = loss + output[i, target_category[i]]
            return loss
        elif method == 'reg':
            # for regression task, output.shape = (batsi_size, 1, N_sector)
            for datum_idx in range(output.shape[0]):
                # print(output[datum_idx, 0, target_category].shape)
                loss = loss + output[datum_idx, 0, sector_idx]
            loss = loss.sum()
            return loss
        elif method == 'cls':
            # print(sector_idx)
            # for classification task, output.shape = (batsi_size, N_category, N_sector)
            for datum_idx in range(output.shape[0]):
                # print('output shape: ', output.shape)
                loss = loss + output[datum_idx, target_category[datum_idx], sector_idx]
            loss = loss.sum()
            return loss
        else:
            raise ValueError('Unknown method: ', method)
            
    def get_output_target_dim(self, output, target_category, sector_idx=0, task_type = 'cls'):
        # ic(target_category)
        # Get the target dimenison of output
        if task_type == 'reg':
            # for regression task, output.shape = (batch_size, 1, N_sector), batch_size = 1
            # or (batch_size, 1) for data regression
            # if np.ndim(output)
            # return output[0, 0, sector_idx]
            output_target_dim = output[..., sector_idx]
            # ic(output_target_dim.shape)
            return output_target_dim
        elif task_type == 'cls':
            # for classification task, output.shape = (batch_size, N_category, N_sector), batch_size = 1
            # return output[0, target_category[0], sector_idx]
            # print(output.shape)
            if np.ndim(output) == 3:
                return output[0, target_category[0], sector_idx]
            elif np.ndim(output) == 2:
                return output[0, sector_idx]
            elif np.ndim(output) == 1:
                return output[sector_idx]
        else:
            raise ValueError('Unknown task type: ', task_type)
        
    
    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        # print('weights:', weights)
        # ic(weights)
        weighted_activations = weights[:, :, None, None] * activations
        # ic(weighted_activations.shape)
        # ic(weighted_activations[...,60, 20:23])
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        # ic(cam[...,60, 20:23])
        return cam

    # def forward_OLD(self, input_tensor, output_type:str='cls', target_category=None, sector_idx:int=0, eigen_smooth=False, counter_factual=False):
    #     # for regression task, output.shape = (batch_size, 1, N_sector)
    #     # for classification task, output.shape = (batch_size, N_category, N_sector)
    #     # def change_precision(data, cuda):
    #         # if cuda:
    #         #     data = torch.cuda.FloatTensor(data)
    #         # else:
    #         #     data = torch.FloatTensor(data)
    #         # return data
        
    #     if self.cuda:
    #         input_tensor = input_tensor.cuda()
        
    #     # output = change_precision(self.activations_and_grads(input_tensor), self.cuda)
    #     output = self.activations_and_grads(input_tensor)
            
        
    #     # print(output.shape)

    #     if type(target_category) is int:
    #         target_category = [target_category] * input_tensor.size(0)
                
        
    #     if output_type in ['cls']:
    #         if target_category is None:
    #             target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
    #         else:
    #             assert(len(target_category) == input_tensor.size(0))
    #     else:
    #         target_category = [0] * input_tensor.size(0)

    #     self.model.zero_grad()
        
    #     # loss = self.get_loss(output, target_category, sector_idx, output_type)
    #     output_sum = self.get_output_sum(output, target_category, sector_idx, output_type)
    #     # print('loss.shape: ', loss.shape)
    #     output_sum.backward(retain_graph=True)

    #     activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
    #     grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

    #     cam = self.get_cam_image(input_tensor, target_category, 
    #         activations, grads, eigen_smooth)
        
    #     # cam = change_precision(cam, cuda=False)
    #     # cam = np.float32(cam)
    #     if counter_factual:
    #         cam = np.maximum(-cam, 0)
    #     else:
    #         cam = np.maximum(cam, 0)

    #     result = []
    #     # print(input_tensor.shape)
    #     for img in cam:
    #         # img = np.float32(img)
    #         # print(img.shape)            
    #         # img = cv2.resize(img, input_tensor.shape[-2:][::-1])
    #         # print(type(img))
    #         # print(img.dtype)
    #         img = skresize(img, input_tensor.shape[-2:])
    #         img = img - np.min(img)
    #         img = img / np.max(img)
    #         result.append(img)
    #     result = np.float32(result)
    #     # print(result.shape)
        # return result
    
    def forward(self, input_datum:dict, input_info:list, output_info:list, device, \
                task_type:str='cls', \
                target_category=-1, sector_idx:int=0, \
                eigen_smooth=False, counter_factual=False,\
                evaluation_config=None):
        # for regression task, output.shape = (batch_size, 1, N_sector)
        # for classification task, output.shape = (batch_size, N_category, N_sector)
        
        input_types = [term['type'] for term in input_info]
        output_types = [term['type'] for term in output_info]
        
        
        # Now don't support batch input
        if type(target_category) is int:
            target_category = [target_category]# * len(datum)
            
        input_strainmat_type = get_data_type_by_category('strainmat', input_types)
        strainmat_squeeze_shape = np.squeeze(input_datum[input_strainmat_type]).shape
        
        # 1. Add activation and gradient hooks to target layer
        #    Pass forward to get output
        datum_eval = {}
        for key in input_types + output_types:
            # datum_eval[key] = torch.from_numpy(datum[key][None,:]).to(self.device)
            # datum_eval[key] = torch.from_numpy(input_datum[key]).to(device, dtype=torch.half)
            datum_eval[key] = torch.from_numpy(input_datum[key]).to(device)
        output = self.activations_and_grads(datum_eval)
        # output_numpy_dict = {}
        # for key in output.keys():
        #     # print(key)
        #     output_numpy_dict[key] = copy.deepcopy(output[key].cpu().data.numpy())
        activations = copy.deepcopy(self.activations_and_grads.activations[-1].cpu().data.numpy())
        datum_eval['net_output'] = output
        
        
        # print(total_loss.dtype)
        # 3. Get activation map and gradient for each type of output
        cams = {}
        # for output_type in output_types:
        # for curr_output_info in output_info:
        for curr_output_info in [output_term for output_term in output_info if output_term['tag'] in ['reg']]:
            output_type = curr_output_info['type']
            # ic(output_type)
            self.model.zero_grad()
            self.activations_and_grads.clear()
            # current_cam = {'type': output_type}
            curr_output = output[output_type]
            # curr_output = output_numpy_dict[output_type]
            curr_output_target_dim = self.get_output_target_dim(curr_output, target_category, sector_idx=sector_idx, task_type=task_type)
            # ic(curr_output_target_dim)
            curr_output_target_dim.backward(retain_graph=True)
            # print(self.activations_and_grads.gradients)
            curr_grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()
            # print(curr_grads.shape)
            # ic(curr_grads[..., 60, 20:25])
    
            curr_cam_raw = self.get_cam_image(datum_eval[input_strainmat_type], target_category, 
                activations, curr_grads, eigen_smooth)
            
            curr_cam = np.maximum(curr_cam_raw, 0)
            curr_counter_cam = np.maximum(-curr_cam_raw, 0)
            # if counter_factual:
            #     curr_cam = np.maximum(-curr_cam, 0)
            # else:
            #     curr_cam = np.maximum(curr_cam, 0)
                
            # current_cam['cam'] = curr_cam
            # current_cam['counter_cam'] = curr_counter_cam
            # cams.append(current_cam)
            # ic(curr_cam.shape)
            # ic(curr_counter_cam.shape)
            cams[output_type] = {'cam': curr_cam, 'counter_cam': curr_counter_cam}
        
        # 4. Get activation map and gradient for each type of total loss (sub-rask loss later)
        self.model.zero_grad()
        self.activations_and_grads.clear()
        # ic(total_loss.dtype)
        # return None
        # 2. Compute loss
        # total_loss, _ = evaluate(datum_eval, output_types, evaluation_config['method'], evaluation_config['paras'])
        
        # total_loss, _ = evaluate(data = datum_eval, 
        #                         data_types_to_eval = output_types,
        #                         output_info = output_info,
        #                         method = evaluation_config['method'], 
        #                         paras = evaluation_config['paras'])
        # total_loss.backward(retain_graph=True)
        # total_loss_grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()
        # # ic(np.sum(total_loss_grads - curr_grads))
        # # ic(all(np.sum(total_loss_grads == curr_grads)))
        # total_loss_cam_raw = self.get_cam_image(datum_eval, target_category, 
        #         activations, total_loss_grads, eigen_smooth)
        
        
        # total_loss_cam = np.maximum(total_loss_cam_raw, 0)
        # total_loss_counter_cam = np.maximum(-total_loss_cam_raw, 0)        
        # # ic(np.sum(np.abs(total_loss_cam - curr_cam)))
        # # cams.append({'type': 'total_loss', 
        # #              'cam': total_loss_cam,
        # #              'counter_cam': total_loss_counter_cam})
        # # ic(total_loss_cam.shape)
        # cams['total_loss'] = {'cam': total_loss_cam, 'counter_cam': total_loss_counter_cam}
        
        # result = []
        for cam_target in cams.keys():
            cam = cams[cam_target]
            for cam_key in ['cam', 'counter_cam']:
                img = cam[cam_key]
                # print(np.min(img), np.max(img))
                img = skresize(np.squeeze(img), strainmat_squeeze_shape[-2:])
                img = img - np.min(img)
                if np.abs(np.max(img)) >= 1e-20:
                    
                    # ic(cam['cam'].shape)
                    
                    # ic(img.shape)
                    # img = img - np.min(img)
                    img = img / np.max(img)
                    pass
                # cam[cam_key] = img
                cams[cam_target][cam_key] = img
        return cams

    
    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam
    # def __call__OLD(self,
    #              input_tensor,
    #              output_type='cls',
    #              target_category=None,
    #              sector_idx:int=0,
    #              aug_smooth=False,
    #              eigen_smooth=False,
    #              counter_factual=False):
    #     if aug_smooth is True:
    #         return self.forward_augmentation_smoothing(input_tensor,
    #             target_category, eigen_smooth, counter_factual)

    #     return self.forward(input_tensor,output_type,
    #         target_category, sector_idx, eigen_smooth, counter_factual)
    
    def __call__(self,
                 input_datum:dict, 
                 input_info:list, 
                 output_info:list, 
                 device,
                 task_type='cls',
                 target_category=None,
                 sector_idx:int=0,
                 aug_smooth=False,
                 eigen_smooth=False,
                 counter_factual=False,
                 evaluation_config=None):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_datum,
                target_category, eigen_smooth, counter_factual)
        
        return self.forward(input_datum=input_datum, 
                            input_info=input_info, 
                            output_info=output_info, 
                            device=device,
                            task_type=task_type, 
                            target_category=target_category, 
                            sector_idx=sector_idx, 
                            eigen_smooth=eigen_smooth, 
                            counter_factual=counter_factual,
                            evaluation_config=evaluation_config)        