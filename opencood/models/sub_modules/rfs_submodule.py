# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet101
import torch.nn.functional as F
from opencood.utils.camera_utils import bin_depths
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fuse_modules.fusion_in_one import \
    MaxFusion, AttFusion, V2VNetFusion, V2XViTFusion, DiscoFusion

class EfficientNetFeatureExtractor(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, output_channels, target_H, target_W):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征
        self.features = nn.Sequential(self.trunk.features[:5]) #sequential 5 layers will be used
        self.conv = nn.Conv2d(in_channels = 112, out_channels=output_channels, kernel_size=1) #112 due to the layers' setting of efficientnet
        self.upsample = nn.Upsample(size=(target_H, target_W), mode='bilinear', align_corners=False)

    def forward(self, x):
        N, M, C, H, W = x.size()
        x = x.view(-1, C, H, W) #resizing to N*M C H W
        x = self.features(x) #feature extraction
        x = self.conv(x) #channel settings
        x = self.upsample(x) #upsamling to target size
        return x

class ResNetFeatureExtractor(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, output_channels, target_H, target_W):
        super(ResNetFeatureExtractor, self).__init__()
        self.trunk = resnet101(pretrained=False, zero_init_residual=True)  # 使用 resnet101 提取特征
        self.features = nn.Sequential(*list(self.trunk.children())[:-3]) #sequential 5 layers will be used
        self.conv = nn.Conv2d(in_channels = 1024, out_channels=output_channels, kernel_size=1) #1024 due to the layers' setting of resnet101
        self.upsample = nn.Upsample(size=(target_H, target_W), mode='bilinear', align_corners=False)

    def forward(self, x):
        N, M, C, H, W = x.size()
        x = x.view(-1, C, H, W) #resizing to N*M C H W
        x = self.features(x) #feature extraction
        x = self.conv(x) #channel settings
        x = self.upsample(x) #upsamling to target size
        return x
