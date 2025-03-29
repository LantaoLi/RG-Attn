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

from torchinfo import summary

def viewer_parser():
    parser = argparse.ArgumentParser(description="for viewing model structures")
    parser.add_argument("--path", "-p", default='opencood/logs/HEAL_m1_based/final_infer',
                        help='Please specify the directory of the model!')
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

def main():
    opt = viewer_parser()
    base_model_path = opt.path
    model_0 = load_model(base_model_path)
    model_0.eval()
    print(model_0)

if __name__ == '__main__':
    main()
