# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics

import torch
import torch.nn as nn
import numpy as np
import yaml
import re

from opencood.tools import train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils

def comparing_parser():
    parser = argparse.ArgumentParser(description="for comparing potentail model difference")
    parser.add_argument("--method", "-m", default="euclidean",
                        help='euclidean, covariance or kl, choose one please.')
    opt = parser.parse_args()
    return opt

def load_model(file_path):
    config_path = os.path.join(file_path, 'config.yaml')
    param = yaml_utils.load_yaml(config_path, None)
    model = train_utils.create_model(param)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_epoch, model = train_utils.load_saved_model(file_path, model)
    print('Loading Model from checkpoint:' + str(resume_epoch))
    return model

def euclidean_distance(base_model, compare_model, comparing_part):
    distance = 0.0
    for (name1, param1), (name2, param2) in zip(base_model.named_parameters(), compare_model.named_parameters()):
        if (comparing_part == 'all' or comparing_part in name1):
            distance += torch.sum((param1 - param2)**2).item()
    return np.sqrt(distance)

def covariance_difference(base_model, compare_model, comparing_part):
    cov_diff = 0.0
    for (name1, param1), (name2, param2) in zip(base_model.named_parameters(), compare_model.named_parameters()):
        if comparing_part == 'all' or comparing_part in name1:
            param1_flat = param1.view(-1).cpu().detach().numpy()
            param2_flat = param2.view(-1).cpu().detach().numpy()
            if len(param1_flat) > 1:
                cov1 = np.cov(param1_flat)
                cov2 = np.cov(param2_flat)
                cov_diff += np.sum((cov1 - cov2)**2)
    return cov_diff

def kl_divergence(base_model, compare_model, comparing_part):
    kl_div = 0.0
    epsilon = 1e-10
    for (name1, param1), (name2, param2) in zip(base_model.named_parameters(), compare_model.named_parameters()):
        if comparing_part == 'all' or comparing_part in name1:
            param1_flat = param1.view(-1).cpu().detach().numpy() + epsilon
            param2_flat = param2.view(-1).cpu().detach().numpy() + epsilon
            p = np.abs(param1_flat)/np.sum(np.abs(param1_flat))
            q = np.abs(param2_flat)/np.sum(np.abs(param2_flat))
            kl_div += np.sum(p * np.log(p/q))
    return kl_div

def main():
    """
    comparison_models = ['opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_2',
        'opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_3',
        'opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_4',
        'opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_5']
    """
    comparison_models = ['opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_2']
    base_model = 'opencood/logs/HEAL_m1_based/stage3/m2_80partial_trained_1'
    model_0 = load_model(base_model)
    model_0.eval()
    #print(model_0)
    lowest_euclidean = None
    lowest_covariance = None
    lowest_kl = None
    for comparison_model in comparison_models:
        model_1 = load_model(comparison_model)
        model_1.eval()
        e = euclidean_distance(model_0, model_1, 'pyramid_backbone.resnet.layer0.0.conv1')
        print("e diff at: " + str(e))
        c = covariance_difference(model_0, model_1, 'pyramid_backbone.resnet.layer0.0.conv1')
        print("c diff at: " + str(c))
        kl = kl_divergence(model_0, model_1, 'pyramid_backbone.resnet.layer0.0.conv1')
        print("kl diff at: " + str(kl))
        if comparison_model == comparison_models[0]:
            lowest_euclidean = [comparison_model, e]
            lowest_covariance = [comparison_model, c]
            lowest_kl = [comparison_model, kl]
        else:
            if e < lowest_euclidean[1]:
                lowest_euclidean = [comparison_model, e]
            if c < lowest_covariance[1]:
                lowest_covariance = [comparison_model, c]
            if kl < lowest_kl[1]:
                lowest_kl = [comparison_model, kl]
    print(lowest_euclidean)
    print(lowest_covariance)
    print(lowest_kl)

if __name__ == '__main__':
    main()
