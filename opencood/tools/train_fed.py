# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import copy

import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools.fed.fed_client import FedClient
from opencood.tools.fed.fed_server import FedServer

from icecream import ic
import numpy as np


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--partial_training', '-p', type=float, default=1.0,
                        help='the partial percentage of training dataset.')
    parser.add_argument('--agent_num', '-n', type=int, default=2,
                        help='the number of agents for so-called federated learning.')
    parser.add_argument('--epoches_per_agr', '-e', type=int, default=1,
                        help='the number epoches per aggregation.')
    parser.add_argument('--mini_train', '-m', type=bool, default=False,
                        help='whether enable mini train.')
    parser.add_argument('--true_single', '-t', type=bool, default=False,
                        help='whether enable agent train with single agent data.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    saved_path = opt.model_dir
    #to Initialize all the clients
    client_list = []
    #client_name_list = [] #just record ids of clients
    for n in range(opt.agent_num):
        client_list.append(FedClient(os.path.join(saved_path, 'Client%s'%(str(n))), str(n), hypes['train_params'].get("single_weight", 1)))

    fed_server = FedServer(client_list)
    fed_server.agg_layer_list = hypes['agg_layer_list']

    print('Dataset Building:')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    if opt.true_single == True:
        ego_hypes = copy.deepcopy(hypes)
        ego_hypes['reverse_ego'] = False
        fixed_opencood_train_dataset = build_dataset(ego_hypes, visualize=False, train=True)
    train_sampler = None
    for client_i in client_list:
        #random sampling the training data according to the partial percentage
        client_i.n_data = len(opencood_train_dataset)
        if opt.partial_training <= 1.0:
            train_size = int(opt.partial_training * len(opencood_train_dataset))
            client_i.n_data = train_size
            indices = list(range(len(opencood_train_dataset)))
            np.random.shuffle(indices)
            client_i.train_indices = indices[:train_size]
            train_sampler = SubsetRandomSampler(client_i.train_indices)
        if opt.true_single == True and client_list.index(client_i) == 0:
            client_i.train_loader = DataLoader(fixed_opencood_train_dataset,
                                      batch_size=hypes['train_params']['batch_size'],
                                      num_workers=4,
                                      collate_fn=opencood_train_dataset.collate_batch_train,
                                      #shuffle=True,
                                      pin_memory=True,
                                      drop_last=True,
                                      prefetch_factor=2,
                                      sampler = train_sampler)
        else:
            client_i.train_loader = DataLoader(opencood_train_dataset,
                                      batch_size=hypes['train_params']['batch_size'],
                                      num_workers=4,
                                      collate_fn=opencood_train_dataset.collate_batch_train,
                                      #shuffle=True,
                                      pin_memory=True,
                                      drop_last=True,
                                      prefetch_factor=2,
                                      sampler = train_sampler)
        client_i.val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)
        print('Dataset Building finished for client ' + client_i.client_name)

    print('Creating Models:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # server model config
    fed_server.model = train_utils.create_model(hypes)
    print(fed_server.model)
    fed_server.device = device
    # client model config
    for client_i in client_list:
        client_i.model = train_utils.create_model(hypes)
        client_i.device = device
        client_i.epoches_per_agr = opt.epoches_per_agr
        # record lowest validation loss checkpoint.
        client_i.lowest_val_loss = 1e5
        client_i.lowest_val_epoch = -1

        # define the loss
        client_i.criterion = train_utils.create_loss(hypes)

        # optimizer setup
        client_i.optimizer = train_utils.setup_optimizer(hypes, client_i.model)
        # lr scheduler setup

        client_i.init_epoch = 0
        # needs 2nd thinking
        client_i.saved_path = client_i.model_dir
        client_i.writer = SummaryWriter(client_i.saved_path)
        client_i.scheduler = train_utils.setup_lr_schedular(hypes, client_i.optimizer)
        client_i.supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
        client_i.eval_freq = hypes['train_params']['eval_freq']
        client_i.save_freq = hypes['train_params']['save_freq']
        client_i.maximum_global_epoch = hypes['train_params']['epoches']
        print('Model creation finished for client ' + client_i.client_name)
        #client_i.print_all_attributes()

    print('Training start')
    epoches = hypes['train_params']['epoches']

    federated_global_dict = fed_server.state_dict()
    #supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate
    total_rounds = int(epoches/opt.epoches_per_agr) # to be used in the next for loop
    for cur_round in range(0, total_rounds): #should be range(0, total_rounds)
        if not opt.mini_train:
            for client_i in client_list:
                #some operations before training
                client_i.update(federated_global_dict)
                #train it!
                state_dict_i, n_data_i, loss_i = client_i.train(cur_round)
                fed_server.rec(client_i.client_name, state_dict_i, n_data_i, loss_i)
                #print(n_data_i, loss_i)
            #aggregation on server
            federated_global_dict, avg_loss, _ = fed_server.agg()
            fed_server.flush()
            # the location of dataset reinitialization could be redesigned
            opencood_train_dataset.reinitialize()
        else:
            total_divide = 4 #temp setting
            for cur_divide in range(total_divide):
                for client_i in client_list:
                    #some operations before training
                    client_i.update(federated_global_dict)
                    start = int(len(client_i.train_indices)/total_divide)*cur_divide
                    end = start + int(len(client_i.train_indices)/total_divide) - 1
                    minitrain_indices = client_i.train_indices[start:] if cur_divide == total_divide - 1 else client_i.train_indices[start:end]
                    minitrain_sampler = SubsetRandomSampler(minitrain_indices)
                    client_i.minitrain_loader = DataLoader(opencood_train_dataset,
                                              batch_size=hypes['train_params']['batch_size'],
                                              num_workers=4,
                                              collate_fn=opencood_train_dataset.collate_batch_train,
                                              #shuffle=True,
                                              pin_memory=True,
                                              drop_last=True,
                                              prefetch_factor=2,
                                              sampler = minitrain_sampler)
                    #train it!
                    state_dict_i, n_data_i, loss_i = client_i.mini_train(cur_round, int(start/hypes['train_params']['batch_size']), cur_divide==total_divide-1)
                    fed_server.rec(client_i.client_name, state_dict_i, n_data_i, loss_i)
                    #print(n_data_i, loss_i)
                #aggregation on server
                federated_global_dict, avg_loss, _ = fed_server.agg()
                fed_server.flush()
            # the location of dataset reinitialization could be redesigned
            opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved!')

if __name__ == '__main__':
    main()
