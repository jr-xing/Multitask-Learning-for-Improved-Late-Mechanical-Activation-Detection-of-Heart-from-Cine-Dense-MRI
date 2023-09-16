# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:25:50 2021

@author: remus
"""

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
# from modules.networks.losses import get_loss
import numpy as np
# https://github.com/ipython/ipython/issues/10627
import matplotlib;

matplotlib.use('agg')
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from modules.evaluation import evaluate
from modules.regularization import regularize
import yaml


def weights_init(m):
    # https://discuss.pytorch.org/t/reset-model-weights/19180/3
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
    # torch.nn.init.xavier_uniform(m.weight)


class NetModule(object):
    def __init__(self, network=None, evaluation_config=None, regularization_config=None, device=torch.device("cpu")):
        # Set network structure
        self.device = device
        self.network = network
        self.evaluation_config = evaluation_config
        self.regularization_config = regularization_config
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-5,
                                              weight_decay=1e-5)
        # self.continueTraining = False

    def train(self, training_dataset, training_config, valid_dataset=None,
              NNI=False, logger=None, init_weights=True,
              evaluation_config=None, regularization_config=None,
              device=None,
              early_stop=True, enable_autocast = True):
        
        
        # Check network
        if self.network is None:
            print('No network assigned yet!')
            return

        # Set network input types
        # self.network.set_input_types(training_dataset.input_types)
        # self.network.set_output_types(training_dataset.output_types)

        # Check evaluation config
        if evaluation_config is None and self.evaluation_config is None:
            print('No evalutation config assigned yet!')
            return
        elif evaluation_config is None and self.evaluation_config is not None:
            evaluation_config = self.evaluation_config

        # Set regularization config
        if regularization_config is None and self.regularization_config is not None:
            regularization_config = self.regularization_config

        # Set device
        device = device if device is not None else self.device
        # self.network.set_device(device)

        # Create dataloader        
        training_dataloader = DataLoader(training_dataset, batch_size=training_config['batch_size'],
                                         shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=training_config['batch_size'],
                                      shuffle=True)

        # Set Optimizer
        if init_weights == True:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=training_config['learning_rate'],
                                              weight_decay=1e-5)

        # Save valid image if needed
        ifValid = training_config.get('valid_check', False) and valid_dataset is not None

        # Prepare validation data
        # if ifValid:
        #     valid_data = valid_dataset.data
        #     for datum in valid_data:
        #         # print(type(datum))
        #         for key in valid_dataset.input_types + valid_dataset.output_types:
        #             if type(datum[key]) is np.ndarray:
        #                 # datum[key] = torch.from_numpy(datum[key]).to(device, dtype = torch.float) 
        #                 datum[key] = torch.from_numpy(datum[key]).to(device) 

        # Initalize if not continue previosu training
        if init_weights == True:
            self.network.apply(weights_init)

        # Creates a GradScaler for mixed precision training
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        self.scaler = GradScaler(enabled=True)

        # Check NNI
        if NNI:
            import nni

        # Training process
        start_time = time.time()
        training_loss_history = np.zeros([0])
        valid_loss_history = np.zeros([0])

        # variables for early stop
        n_early_stop_wait_epoch_valid = 0
        n_early_stop_wait_epoch_valid_patience = 5
        n_early_stop_wait_epoch_training = 0
        n_early_stop_wait_epoch_training_patience = 5
        valid_loss_best = 1e10
        training_loss_best = 1e10
        print('Start training!')
        # torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(1, training_config['epochs_num'] + 1):
            output_has_nan = False
            # print('epoch ', epoch)
            for datum in training_dataloader:
                self.optimizer.zero_grad()
                # print('Batch loaded...')
                for key in datum.keys():
                    # datum[key].to(self.device, dtype = torch.float)
                    datum[key] = datum[key].to(device)
                    # print(key, datum[key].dtype)
                # datum[key].cuda()
                # datum_input = [datum[key].to(self.device, dtype = torch.float) for key in training_dataset.input_types]
                # datum_output = self.network(datum_input)
                with autocast(enabled = enable_autocast):
                    # datum['output'] = self.network(datum_input)
                    datum['net_output'] = self.network(datum)
                    for key in datum['net_output'].keys():
                        # print(type(datum['net_output'][key]))
                        if torch.isnan(datum['net_output'][key]).any():
                            output_has_nan = True
                # if output_has_nan:
                #     # print('NAN!')
                #     continue
                    
                    training_loss, training_reg_loss = evaluate(data = datum, 
                                                                data_types_to_eval = training_dataset.output_types,
                                                                output_info = training_dataset.output_info,
                                                                method = evaluation_config['method'], 
                                                                paras = evaluation_config['paras'])
                    # training_loss += regularize(self.network, regularization_config['method'], regularization_config['paras'], device)

                # ===================backward====================                
                
                if not enable_autocast:
                    training_loss.backward()                
                    self.optimizer.step()

                # print(training_loss.dtype)
                # training_loss = training_loss.half()
                # print(training_loss.dtype)
                # print(training_loss)
                else:
                    self.scaler.scale(training_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
            report_epochs_num = training_config.get('try_early_stop_per_epochs', 10)
            # Try Early Stop
            # https://github.com/ncullen93/torchsample/blob/master/torchsample/callbacks.py
            if early_stop and epoch % report_epochs_num == 0 and epoch > 50 and training_loss <= 20:
                if ifValid:
                    valid_loss, valid_reg_loss = self.validate(valid_dataloader = valid_dataloader, 
                                                               output_info = valid_dataset.output_info,
                                                               evaluation_config = evaluation_config, 
                                                               device = device)
                    if (valid_loss - valid_loss_best) / valid_loss_best <= 0:
                        valid_loss_best = valid_loss
                        n_early_stop_wait_epoch_valid = 0
                    elif (valid_loss - valid_loss_best) / valid_loss_best < 1e-3:
                        n_early_stop_wait_epoch_valid = 0
                    else:
                        n_early_stop_wait_epoch_valid += 1
                    if n_early_stop_wait_epoch_valid >= n_early_stop_wait_epoch_valid_patience:
                        print(f'Triggered Early step at epoch {epoch}')
                        print(f'Curr loss: {valid_loss}, best loss: {valid_loss_best}')
                        # if epoch < 100:
                        #     epoch = 0                        
                        break

                if (training_loss - training_loss_best) / training_loss_best <= 0:
                    training_loss_best = valid_loss
                    n_early_stop_wait_epoch_training = 0
                elif (training_loss - training_loss_best) / training_loss_best < 1e-1:
                    n_early_stop_wait_epoch_training = 0
                else:
                    n_early_stop_wait_epoch_training += 1
                if n_early_stop_wait_epoch_training >= n_early_stop_wait_epoch_training_patience:
                    print(f'Triggered Early step at epoch {epoch}')
                    print(f'Curr loss: {valid_loss}, best loss: {valid_loss_best}')
                    # if epoch < 100:
                    #     epoch = 0                        
                    break

            # ===================log========================            
            if epoch % report_epochs_num == 0:
                # allloss = self.pred(alldata, alllabels, avg = True)[1]
                # Report Time and Statistics
                # loss_history = np.append(loss_history, loss.to(torch.device('cpu')).detach().numpy())
                if ifValid:
                    valid_loss, valid_reg_loss = self.validate(valid_dataloader = valid_dataloader, 
                                                               # output_info = valid_dataset.output_types,
                                                               output_info = valid_dataset.output_info,
                                                               evaluation_config = evaluation_config, 
                                                               device = device)
                    
                    valid_loss_str = f'Avg lossVa:{valid_loss:.4E}, '
                    # Break if NaN of Inf
                    if torch.isnan(valid_loss) or torch.isinf(valid_loss):                        
                        print('Stop training because valid loss = ', valid_loss)
                        valid_loss, training_loss = 4e3, 4e3
                        break
                else:
                    valid_loss = np.nan
                    valid_loss_str = ''
                valid_loss_history = np.append(valid_loss_history, valid_loss.to(torch.device('cpu')).detach().numpy())
                # validLoss_history = torch.cat((validLoss_history, validLoss))
                past_time = time.time() - start_time
                time_per_epoch_min = (past_time / epoch) / 60
                # logger.info('Final accuracy reported: %s', accuracy)

                # print(f'valid_loss.item(): {type(valid_loss.item())}')
                if NNI:
                    nni_report = {'default': valid_reg_loss.item(), 'valid_loss': valid_loss.item(),
                                  'training_loss': training_loss.item()}
                    nni.report_intermediate_result(nni_report)

                print(f'epoch [{epoch:3d}/{training_config["epochs_num"]:3d}], ' +
                      f'Avg lossTr:{training_loss:.4E}, ' +
                      valid_loss_str +
                      f'used: {past_time / 60:.1f} mins, ' +
                      f'finish in:{(training_config["epochs_num"] - epoch) * time_per_epoch_min:.0f} mins')
                # logger.info('Final accuracy reported: %s', valid_loss)
                # logger.debug('test accuracy %g', valid_loss)
                # logger.debug('Pipe send intermediate result done.')

        training_loss_final = training_loss if type(training_loss) is float else training_loss.item()
        valid_loss_final = valid_loss if type(valid_loss) is float else valid_loss.item()
        valid_reg_loss_final = valid_reg_loss if type(valid_reg_loss) is float else valid_reg_loss.item()

        if NNI:
            nni_report = {'default': valid_reg_loss_final, 'valid_loss': valid_loss_final,
                          'training_loss': training_loss_final}
            nni.report_final_result(nni_report)
            logger.info(f'Final valid loss {valid_loss}')

        past_time = time.time() - start_time
        print(f'Traing finished with {training_config["epochs_num"]} epochs and {past_time / 3600} hours')
        return training_loss_final, training_loss_history, valid_loss_final, valid_loss_history, valid_reg_loss.item(), past_time

    # def validate(self, valid_data, output_types, evaluation_config, regularization_config, device):
    #     valid_loss = 0
    #     for datum in valid_data:
    #         datum['net_output'] = self.network(datum)
    #         valid_loss += evaluate(datum, output_types, evaluation_config['method'], evaluation_config['paras'])
    #     if regularization_config is not None:
    #         valid_loss += regularize(self.network, regularization_config['method'], regularization_config['paras'], device)
    #     return valid_loss

    def validate(self, valid_dataloader, output_info, evaluation_config, device):
        valid_loss = 0
        valid_reg_loss = 0
        for datum in valid_dataloader:
            for key in datum.keys():
                datum[key] = datum[key].to(device)
            with autocast():
                datum['net_output'] = self.network(datum)
                # for kkey in datum['net_output'].keys():
                #     print(kkey, datum['net_output'][kkey].get_device(), datum[kkey].get_device())
                datum_valid_loss, datum_valid_reg_loss = evaluate(data = datum, 
                                                                  data_types_to_eval = [info['type'] for info in output_info], 
                                                                  output_info = output_info, 
                                                                  method = evaluation_config['method'],
                                                                  paras = evaluation_config['paras'])
                valid_loss += datum_valid_loss
                valid_reg_loss += datum_valid_reg_loss
        return valid_loss, valid_reg_loss

    def pred_OLD(self, data, label=None, evaluation_config=None):
        evaluation_config = self.evaluation_config if evaluation_config is None else evaluation_config
        loss = None
        if type(data) is np.ndarray and type(label) is np.ndarray:
            data_output = self.network({'data': data})
            if label is not None:
                data_to_eval = {'output': data_output, 'label': label}
                loss = evaluate(data_to_eval, evaluation_config['method'], evaluation_config['paras'])
            return data_output, loss
        elif type(data) is dict:
            data_output = self.network(data)
            if label is not None:
                data['output'] = data_output
                loss = evaluate(data, evaluation_config['method'], evaluation_config['paras'])
            return data, loss

    # @autocast()
    def pred(self, datum: dict, input_types: list, output_info: list, enable_autocast=True, evaluate=True, evaluation_config=None):
        # if type(data) is dict:
        #     data = [data]

        if evaluation_config is None and self.evaluation_config is None:
            print('No evalutation config assigned yet!')
            return
        elif evaluation_config is None and self.evaluation_config is not None:
            evaluation_config = self.evaluation_config
            
        output_types = [term['type'] for term in output_info]

        # for datum in data:
        # datum_eval = [{key: torch.from_numpy(datum[key][None,:]).to(self.device)} for key in input_types + output_types][0]
        datum_eval = {}
        for key in input_types + output_types:
            # datum_eval[key] = torch.from_numpy(datum[key][None,:]).to(self.device)
            if enable_autocast:
                datum_eval[key] = torch.from_numpy(datum[key]).to(self.device, dtype=torch.half)
            else:
                datum_eval[key] = torch.from_numpy(datum[key]).to(self.device)
        # print(datum.keys())
        # print(input_types + output_types)
        # print(datum_eval.keys())
        with autocast(enabled=enable_autocast):
            # print(type(datum_eval))
            # for key in datum_eval.keys():
            #     print(key, datum_eval[key].dtype)
            datum_eval['net_output'] = self.network(datum_eval)
        
        if evaluate:
            datum_loss, _ = evaluate(data = datum_eval, 
                                    data_types_to_eval = output_types,
                                    output_info = output_info,
                                    method = evaluation_config['method'], 
                                    paras = evaluation_config['paras'])
        else:
            datum_loss = None

        # Convert output
        # for key in datum_eval['net_output'].keys():
        #     if key in ['late_acti_label']:
        #         datum_eval['net_output'][key] = torch.argmax(datum_eval['net_output'][key], axis=1) == 0

        for key in datum_eval['net_output'].keys():
            if type(datum_eval['net_output'][key]) is torch.Tensor:
                datum_eval['net_output'][key] = datum_eval['net_output'][key].to(torch.device('cpu')).detach().numpy()
        return {
            'data': datum_eval['net_output'],
            'loss': datum_loss.item() if datum_loss is not None else None,
            'loss_var': datum_loss
        }
        # datum['output'], datum['loss'] =
        # for key in datum.keys():
        #         # datum[key].to(self.device, dtype = torch.float)
        #          datum[key] = datum[key].to(device)
        # if type(data) is list:
        #     output = self.pred(data)
        # elif type(data)

    # def predOLD(self, data, labels = None):
    #     if type(data) == np.ndarray:
    #         data = torch.from_numpy(data.copy())
    #     if type(labels) == np.ndarray:
    #         labels = torch.from_numpy(labels.copy())
    #     data = data.to(self.device, dtype = torch.float)            

    #     predictions = self.net(data)
    #     #.to(torch.device('cpu')).detach().numpy()
    #     if type(predictions) is tuple:
    #         predictions = list(predictions)
    #         for idx in range(len(predictions)):
    #             predictions[idx] = predictions[idx].to(torch.device('cpu')).detach().numpy()
    #     else:
    #         predictions = predictions.to(torch.device('cpu')).detach().numpy()
    #     if labels is not None:
    #         labels = labels.to(self.device, dtype = torch.float)
    #         # print(self.net(data).shape, labels.shape)
    #         loss = self.criterion(self.net(data), labels)
    #         loss = loss.to(torch.device('cpu')).detach().numpy()            
    #     else:
    #         loss = None
    #     return predictions, loss

    # def pred_joint(self, data):            
    def saveLossHistory(self, loss_history, save_filename, report_epochs_num):
        # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib        
        plt.ioff()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(np.arange(1, (len(loss_history) + 1)) * report_epochs_num, np.log(loss_history))
        fig.savefig(save_filename, bbox_inches='tight')  # save the figure to file
        plt.close(fig)

    def saveLossHistories(self, loss_histories, save_filename, report_epochs_num=1, legends=None):
        # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib        
        plt.ioff()
        fig, axe = plt.subplots(nrows=1, ncols=1)

        for idx, loss_history in enumerate(loss_histories):
            line, = axe.plot(np.arange(1, (len(loss_history) + 1)) * report_epochs_num, loss_history)
            if legends is not None:
                line.set_label(legends[idx])
        axe.legend()
        # axe.plot(np.arange(1,(len(loss_history)+1))*report_epochs_num, np.log(loss_history))
        fig.savefig(save_filename, bbox_inches='tight')  # save the figure to file
        plt.close(fig)

    #     def valid(self, img, save_filename, config):
    #         # 1. Go through network
    #         img = torch.from_numpy(img).to(self.device, dtype = torch.float)
    #         outimg = self.net(img).to(torch.device('cpu')).detach().numpy()

    #         # 2. Take slice as an image
    # #        if len(np.shape(img)) == 4:
    # #            # if images are 2D image and img has shape [N,C,H,W]
    # #            img_sample = outimg[config.get('index', 0), :]
    # #        elif len(np.shape(img)) == 5:
    # #            # if images are 3D image and img has shape [N,C,D,H,W]
    # #            img_sample_3D = outimg[config.get('index', 0), :]
    # #            slice_axis = config.get('slice_axis',2)
    # #            slice_index = config.get('slice_index',0)
    # #            if slice_index == 'middle': slice_index = int(np.shape(img_sample_3D)[slice_axis]/2)
    # #            if slice_axis == 0:
    # #                img_sample = img_sample_3D[:,slice_index,:,:]
    # #            elif slice_axis == 1:
    # #                img_sample = img_sample_3D[:,:,slice_index,:]
    # #            elif slice_axis == 2:
    # #                img_sample = img_sample_3D[:,:,:,slice_index]
    # #        else:
    # #            raise ValueError(f'Wrong image dimension. \
    # #                             Should be 4 ([N,C,H,W]) for 2d images \
    # #                             and 5 ([N,C,D,H,W]) for 3D images, \
    # #                             but got {len(np.shape(img))}')
    #         img_sample = slice_img(outimg, config)

    #         # 3. Save slice
    #         plt.imsave(save_filename, np.squeeze(img_sample), cmap='gray')

    def save_network(self, filename_full):
        # Save trained net parameters to file
        # torch.save(self.net.state_dict(), f'../model/{name}.pth')
        # torch.save(self.net.state_dict(), filename_full)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename_full)

    def load_network(self, checkpoint_path, map_location=None):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        # Load saved parameters
        self.continueTraining = True
        checkpoint = torch.load(checkpoint_path, map_location)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.network.eval()
        # self.net.load_state_dict(torch.load(model_path))
        # self.net.eval()
