# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistribution

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature

from PIL import Image
import torchvision.transforms.functional as TF
import datetime
import copy
import time

def gray_image_function(input_tensor, image_info):
    gray_tensor = input_tensor.sum(dim=0, keepdim=True)
    gray_tensor = gray_tensor.mul(255).byte()
    gray_image = TF.to_pil_image(gray_tensor)
    now = datetime.datetime.now()
    filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
    gray_image.save(filename)

def color_image_function(input_tensor, image_info): #for 64 channels
    C, H, W = input_tensor.shape
    device = input_tensor.device
    tensor_1 = input_tensor[:22, :, :].mean(dim=0)
    tensor_2 = input_tensor[22:43, :, :].mean(dim=0)
    tensor_3 = input_tensor[43:64, :, :].mean(dim=0)
    color_mapped_tensor = torch.stack([tensor_1, tensor_2, tensor_3], dim=0)
    color_mapped_tensor = color_mapped_tensor.mul(255).byte()
    color_image = TF.to_pil_image(color_mapped_tensor)
    now = datetime.datetime.now()
    filename = f"{image_info}_color_image_{now.strftime('%m%d%H%M%S')}.png"
    color_image.save(filename)

def extract_canvas(bev_map, fov_radians, h1, w1, rots):
    C, H, W = bev_map.shape
    center_x, center_y = W // 2, H // 2
    # angles and rotations
    device = bev_map.device
    half_fov = fov_radians / 2
    angles = torch.linspace(-half_fov.item(), half_fov.item(), w1, device=device) #was -half_fov, half_fov (rotation backwards)
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
    #static width for beam setting
    center_x = torch.tensor(center_x, device = device, dtype=torch.float32)
    center_y = torch.tensor(center_y, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(center_x**2+center_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    x_coords = (center_x + radii*cos_angles).view(h1, w1)
    y_coords = (center_y - radii*sin_angles).view(h1, w1)
    x_coords = x_coords.clamp(0, W-1)/(W-1)*2-1
    y_coords = y_coords.clamp(0, H-1)/(H-1)*2-1
    grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    bev_map = bev_map.unsqueeze(0)
    rectangular_grid = F.grid_sample(bev_map, grid, mode='bilinear', align_corners=True)
    rectangular_grid = rectangular_grid.squeeze(0)
    return rectangular_grid

def restore_canvas(rectangular_grid, bev_map_shape, fov_radians, rots):
    C, H, W = bev_map_shape
    device = rectangular_grid.device
    # Initialize the BEV map and count map
    restore_bev_map = torch.zeros((C, H, W), dtype=rectangular_grid.dtype, device=device)
    count_map = torch.zeros((1, H, W), dtype=rectangular_grid.dtype, device=device)
    _, h1, w1 = rectangular_grid.shape
    center_x, center_y = W // 2, H // 2
    half_fov = fov_radians / 2
    # Precompute angles and radii
    angles = torch.linspace(-half_fov.item(), half_fov.item(), w1, device=device) #was -half_fov, half_fov (rotation backwards)
    # rotation matrix for FOV
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    cos_rots, sin_rots = rot[0,0], rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + +
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was - +
    #static width for beam setting
    center_x = torch.tensor(center_x, device = device, dtype=torch.float32)
    center_y = torch.tensor(center_y, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(center_x**2+center_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    # Compute grid coordinates
    x_coords = (center_x + radii * cos_angles).round().long().clamp(0, W - 1)
    y_coords = (center_y - radii * sin_angles).round().long().clamp(0, H - 1)
    # Flatten the coordinates and values for scatter_add
    x_coords_flat = x_coords.view(-1)
    y_coords_flat = y_coords.view(-1)
    values_flat = rectangular_grid.view(C, -1)
    # Compute linear indices for scatter_add
    indices = y_coords_flat * W + x_coords_flat
    # Scatter add values into restore_bev_map
    restore_bev_map.view(C, -1).scatter_add_(1, indices.expand(C, -1), values_flat)
    count_map.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=count_map.dtype, device=device))
    # Avoid division by zero
    count_map[count_map == 0] = 1
    restore_bev_map /= count_map
    return restore_bev_map

class DepthAwareCrossAttention(nn.Module):
    def __init__(self, model_cfg):
        super(DepthAwareCrossAttention, self).__init__()
        self.C1 = model_cfg["c1"]
        self.C2 = model_cfg["c2"]
        self.num_heads = model_cfg["num_heads"]
        self.h1, self.h2 = model_cfg["h1"], model_cfg["h2"]
        self.dropout = model_cfg["dropout"]
        #qkv linear trans
        self.query_proj = nn.Linear(self.C1, self.C1)
        self.key_proj = nn.Linear(self.C2, self.C1)
        self.value_proj = nn.Linear(self.C2, self.C1)
        self.pos_encoding_a = nn.Parameter(torch.randn(1, self.h1, self.C1))
        self.pos_encoding_b = nn.Parameter(torch.randn(1, self.h2, self.C2))
        self.attention = MultiheadAttention(self.C1, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

    def forward(self, list_a, list_b, fov_list, rots_list, trans_list):
        n = len(list_a)
        if len(list_b) != n or len(fov_list) != n: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            b = list_b[i]
            fov_radians = fov_list[i]
            rots = rots_list[i]
            trans = trans_list[i]
            C1, h1, w1 = a.shape
            C2, h2, w2 = b.shape
            a_rect = extract_canvas(a, fov_radians, h1, w2, rots)
            a_enhanced = torch.zeros_like(a_rect)
            a_rect = a_rect.permute(2,1,0).reshape(w2, h1, C1) #w2=w1, h1, c1
            b_rect = b.permute(2,1,0).reshape(w2, h2, C2) #w2=w1, h2, c2
            a_rect += self.pos_encoding_a
            b_rect += self.pos_encoding_b
            q = self.query_proj(a_rect) #w1h1c1
            k = self.key_proj(b_rect) #w1h2c1
            v = self.value_proj(b_rect) #w1h2c1
            attn_output, _ = self.attention(q, k, v) #w1h1c1
            a_enhanced = attn_output.reshape(w2, h1, C1).permute(2,1,0) #c1h1w1
            a_enhanced = restore_canvas(a_enhanced, a.shape, fov_radians, rots)
            enhanced_a_list.append(a+a_enhanced)
        return enhanced_a_list
