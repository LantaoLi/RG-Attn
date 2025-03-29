# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import copy

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


class FedClient(object):
    def __init__(self, dir, id, single_weight):
        """
        Initialize a client for federated learning.
        Parameters, basically all none for initialization
        """
        self.model_dir = dir
        self.client_name = id
        self.single_weight = single_weight

        self.n_data = 0
        self.train_indices = None
        self.minitrain_loader = None
        self.train_loader = None
        self.val_loader = None

        self.model = None
        self.device = None
        self.epoches_per_agr = None
        self.lowest_val_loss = None
        self.lowest_val_epoch = None
        self.criterion = None
        self.optimizer = None
        self.init_epoch = 0
        # needs 2nd thinking on path
        self.saved_path = None
        self.writer = None
        self.scheduler = None
        self.supervise_single_flag = False
        # frequency of save and evaluation
        self.save_freq = None
        self.eval_freq = None
        self.maximum_global_epoch = 0


    def print_all_attributes(self):
        attributes = vars(self)
        for attr, value in attributes.items():
            print(f"{attr}: {value}")

    def update(self, model_state_dict):
        """
        Update the local model w.r.t. predefined personalization rules.
        Parameters
        ----------
        model_state_dict: dict
            global model state dict from server
        """
        global_model_state_dict = copy.deepcopy(model_state_dict)
        self.model.load_state_dict(global_model_state_dict)

    def train(self, global_round):
        """
        Train model on local dataset.
        Parameters
        ----------
        global_round : int
            current global round number.

        Returns (tensor on cpu rather than gpu) for federating
        -------
        self.model.state_dict() : dict
            Local model state dict.
        self.n_data : int
            Number of local data points.
        final_loss.data.cpu().numpy() : float
            Train loss value.
        """
        self.model.to(self.device)
        #model_without_ddp = self.model
        saved_path = self.model_dir
        # record training
        #writer = SummaryWriter(self.saved_path)
        # used to help schedule learning rate
        for epoch in range(0, 0 + self.epoches_per_agr):
            # to log the real global epoch number
            real_epoch = epoch + global_round*self.epoches_per_agr
            # break if reaching the configured maximum epoches (may be inaccurate at 1 or 2 epoches)
            if real_epoch >= self.maximum_global_epoch:
                break
            self.model.train()
            try: # heter_model stage2
                self.model.model_train_init()
            except:
                print("No model_train_init function")
            #pbar2 = tqdm.tqdm(total=len(self.train_loader), leave=True)

            for i, batch_data in enumerate(self.train_loader):
                if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                    continue
                self.model.zero_grad()
                self.optimizer.zero_grad()
                batch_data = train_utils.to_device(batch_data, self.device)
                # below used to be batch_data['ego']['epoch'] = epoch, modif here to keep epoch increasing
                batch_data['ego']['epoch'] = epoch + self.epoches_per_agr*global_round
                ouput_dict = self.model(batch_data['ego'])

                final_loss = self.criterion(ouput_dict, batch_data['ego']['label_dict'])
                self.criterion.logging(real_epoch, i, len(self.train_loader), self.writer)

                if self.supervise_single_flag:
                    final_loss += self.criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * self.single_weight
                    self.criterion.logging(real_epoch, i, len(self.train_loader), self.writer, suffix="_single")

                # back-propagation
                final_loss.backward()
                self.optimizer.step()

                self.scheduler.step(real_epoch)

            if self.save_freq is not None and real_epoch % self.save_freq == 0:
                self.save_model(real_epoch)
            if self.eval_freq is not None and real_epoch % self.eval_freq == 0 and real_epoch > 1:
                self.eval(real_epoch)

            #self.trainset.reinitialize() #this line should be useless
            #pbar2.update(1)
        self.model.to('cpu')
        return self.model.state_dict(), self.n_data, final_loss.data.cpu().numpy()

    def mini_train(self, global_round, start, finish):
        """
        Train model on local dataset within 1 epoch as mini batch.
        Parameters
        ----------
        global_round : int
            current global round number.

        Returns (tensor on cpu rather than gpu) for federating
        -------
        self.model.state_dict() : dict
            Local model state dict.
        self.n_data : int
            Number of local data points.
        final_loss.data.cpu().numpy() : float
            Train loss value.
        """
        self.model.to(self.device)
        #model_without_ddp = self.model
        saved_path = self.model_dir
        # record training
        #writer = SummaryWriter(self.saved_path)
        # used to help schedule learning rate
        real_epoch = global_round*self.epoches_per_agr
        self.model.train()
        try: # heter_model stage2
            self.model.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(self.minitrain_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            self.model.zero_grad()
            self.optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, self.device)
            # below used to be batch_data['ego']['epoch'] = epoch, modif here to keep epoch increasing
            batch_data['ego']['epoch'] = real_epoch
            ouput_dict = self.model(batch_data['ego'])

            final_loss = self.criterion(ouput_dict, batch_data['ego']['label_dict'])
            self.criterion.logging(real_epoch, i + start, len(self.train_loader), self.writer)

            if self.supervise_single_flag:
                final_loss += self.criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * self.single_weight
                self.criterion.logging(real_epoch, i + start, len(self.train_loader), self.writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            self.optimizer.step()
            self.scheduler.step(real_epoch)
        if finish:
            if self.save_freq is not None and real_epoch % self.save_freq == 0:
                self.save_model(real_epoch)
            if self.eval_freq is not None and real_epoch % self.eval_freq == 0 and real_epoch > 1:
                self.eval(real_epoch)

        self.model.to('cpu')
        return self.model.state_dict(), len(self.minitrain_loader), final_loss.data.cpu().numpy()

    def eval(self, real_epoch):
        """
        Evaluate the local model before federating
        """
        valid_ave_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(self.val_loader):
                if batch_data is None:
                    continue
                self.model.zero_grad()
                self.optimizer.zero_grad()
                self.model.eval()

                batch_data = train_utils.to_device(batch_data, self.device)
                batch_data['ego']['epoch'] = real_epoch
                ouput_dict = self.model(batch_data['ego'])

                final_loss = self.criterion(ouput_dict, batch_data['ego']['label_dict'])
                #print(f'val loss {final_loss:.3f}')
                valid_ave_loss.append(final_loss.item())

        valid_ave_loss = statistics.mean(valid_ave_loss)
        print('At epoch %d, the validation loss is %f' % (real_epoch,
                                                          valid_ave_loss))
        self.writer.add_scalar('Validate_Loss', valid_ave_loss, real_epoch)

        # lowest val loss
        if valid_ave_loss < self.lowest_val_loss:
            self.lowest_val_loss = valid_ave_loss
            torch.save(self.model.state_dict(),
                   os.path.join(self.saved_path,
                                'net_epoch_bestval_at%d.pth' % (real_epoch + 1)))
            if self.lowest_val_epoch != -1 and os.path.exists(os.path.join(self.saved_path,
                                'net_epoch_bestval_at%d.pth' % (self.lowest_val_epoch))):
                os.remove(os.path.join(self.saved_path,
                                'net_epoch_bestval_at%d.pth' % (self.lowest_val_epoch)))
            self.lowest_val_epoch = real_epoch + 1

    def save_model(self, real_epoch):
        """
        Save local model
        """
        torch.save(self.model.state_dict(),
                   os.path.join(self.saved_path,
                                'net_epoch%d.pth' % (real_epoch + 1)))
