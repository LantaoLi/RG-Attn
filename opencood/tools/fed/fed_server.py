# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import argparse
import os
import statistics
import copy

import torch
import tqdm
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, DistributedSampler
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


class FedServer(object):
    def __init__(self, client_list):
        """
        Initialize a server for federated learning.
        Only for avg or similar processing
        No val, test or train
        Parameters
        ----------
        client_list: list
            A list of clients for federated learning.
        """
        self.model = None
        self.device = None

        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}

        self.batch_size = 1

        self.client_list = client_list
        #self.val_loader = None
        self.n_data = 0
        self.round = 0
        self.agg_layer_list = None

    def state_dict(self):
        """

        Returns
        -------

        """
        return self.model.state_dict()

    def agg(self):
        """
        Aggregate the models from clients
        Returns
        -------
        model_state : dict
            global aggregated model state
        avg_loss : float
            average train loss for all clients
        self.n_data : int
            Number of total data points
        """
        client_num = len(self.client_list)
        self.n_data = sum(self.client_n_data.values())
        # such condition should be treated carefully
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        model_state = self.model.state_dict()
        avg_loss = 0.0
        # print('number of selected clients in Cloud: ' + str(client_num))

        #agg_layer_list=['backbone_m1.resnet.layer0.2', 'pyramid_backbone.resnet.layer0.0']

        for i, client_i in enumerate(self.client_list):
            name = client_i.client_name
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if self.agg_layer_list is None or any(string in key for string in self.agg_layer_list):
                    if i == 0:
                        model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                        print(key)
                    else:
                        model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                            name] / self.n_data
            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        return model_state, avg_loss, self.n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        Receive the local models from connected clients.
        Parameters
        ----------
        name : str
            client name.
        state_dict : dict
            uploaded local model from a dedicated client.
        n_data : int
            number of data points in a dedicated client.
        loss : float
            train loss of a dedicated client.
        """
        #self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flush the information for current communication round.
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
