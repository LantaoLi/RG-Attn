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

def gray_image_function(input_tensor, image_info):
    gray_tensor = input_tensor.sum(dim=0, keepdim=True)
    gray_tensor = gray_tensor.mul(255).byte()
    gray_image = TF.to_pil_image(gray_tensor)
    now = datetime.datetime.now()
    filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
    gray_image.save(filename)

def new_fov_bound(fov_radians, affine, H, W, device):
    half_fov = fov_radians / 2
    min_vector = torch.tensor([torch.cos(-half_fov), torch.sin(-half_fov)], device = device)
    max_vector = torch.tensor([torch.cos(half_fov), torch.sin(half_fov)], device = device)
    fov_affine = torch.zeros(affine[:2,:2].shape, device = device, dtype=torch.float32)
    fov_affine = fov_affine + affine[:2,:2]
    fov_affine[0,1] = fov_affine[0,1]*W/H
    fov_affine[1,0] = fov_affine[1,0]*H/W
    transformed_min_vector = torch.matmul(fov_affine[:2,:2], min_vector)
    transformed_max_vector = torch.matmul(fov_affine[:2,:2], max_vector)
    angle_min = -torch.atan2(transformed_min_vector[1], transformed_min_vector[0])
    angle_max = -torch.atan2(transformed_max_vector[1], transformed_max_vector[0])
    new_angle_min = min(angle_min, angle_max)
    new_angle_max = max(angle_min, angle_max)
    if new_angle_min < -torch.pi/2 and new_angle_max > torch.pi/2:
        new_angle_min, new_angle_max = new_angle_max, new_angle_min
        new_angle_min -= 2*torch.pi
    return new_angle_min, new_angle_max

def extract_canvas(bev_map, fov_radians, h1, w1, rots, affine):
    C, H, W = bev_map.shape
    device = bev_map.device
    # config fov center location on BEV
    o_center = torch.tensor([0.0 , 0.0 , 1.0], device = device, dtype=torch.float32)
    affine = affine.to(torch.float32)
    n_center = torch.matmul(affine, o_center)
    center_x, center_y = (n_center[0].item()+1.0)*W//2, (1.0 + n_center[1].item())*H//2
    # angels and rotations
    new_angle_min, new_angle_max = new_fov_bound(fov_radians, affine, H, W, device)
    angles = torch.linspace(new_angle_min, new_angle_max, w1, device=device)
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    # beam slicing
    cos_rots = rot[0,0]
    sin_rots = rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
    # Max radius calculation (fixed)
    half_x = torch.tensor(W//2, device = device, dtype=torch.float32)
    half_y = torch.tensor(H//2, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(half_x**2+half_y**2)
    radii = torch.linspace(0, 1, h1, device=device).view(h1, 1)*radius_max
    #coords
    x_coords = (center_x + radii*cos_angles).view(h1, w1)
    y_coords = (center_y - radii*sin_angles).view(h1, w1)
    x_coords = x_coords.clamp(0, W-1)/(W-1)*2-1
    y_coords = y_coords.clamp(0, H-1)/(H-1)*2-1
    grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)
    bev_map = bev_map.unsqueeze(0)
    rectangular_grid = F.grid_sample(bev_map, grid, mode='bilinear', align_corners=True)
    rectangular_grid = rectangular_grid.squeeze(0)
    return rectangular_grid

def restore_canvas(rectangular_grid, bev_map_shape, fov_radians, rots, affine):
    C, H, W = bev_map_shape
    device = rectangular_grid.device
    # Initialize the BEV map and count map
    restore_bev_map = torch.zeros((C, H, W), dtype=rectangular_grid.dtype, device=device)
    count_map = torch.zeros((1, H, W), dtype=rectangular_grid.dtype, device=device)
    _, h1, w1 = rectangular_grid.shape
    # config fov center location on BEV
    o_center = torch.tensor([0.0 , 0.0 , 1.0], device = device, dtype=torch.float32)
    affine = affine.to(torch.float32)
    n_center = torch.matmul(affine, o_center)
    center_x, center_y = (n_center[0].item()+1.0)*W//2, (1.0 + n_center[1].item())*H//2
    # angels and rotations
    new_angle_min, new_angle_max = new_fov_bound(fov_radians, affine, H, W, device)
    angles = torch.linspace(new_angle_min, new_angle_max, w1, device=device)
    rot = rots[0][:2,:2].to(torch.float32)
    rot = torch.tensor([[0,-1],[1, 0]], device = device, dtype=torch.float32)@rot
    # beam slicing
    cos_rots = rot[0,0]
    sin_rots = rot[1,0]
    cos_angles = cos_rots*torch.cos(angles) + sin_rots*torch.sin(angles) #was + -
    sin_angles = -sin_rots*torch.cos(angles) + cos_rots*torch.sin(angles) #was + +
    # Max radius calculation
    half_x = torch.tensor(W//2, device = device, dtype=torch.float32)
    half_y = torch.tensor(H//2, device = device, dtype=torch.float32)
    radius_max = torch.sqrt(half_x**2+half_y**2)
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

class DepthCocoloring(nn.Module):
    def __init__(self, model_cfg):
        super(DepthCocoloring, self).__init__()
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

    def forward(self, list_a, list_b, affine_matrix, fov_list, rots_list, trans_list):
        n = len(list_a)
        m = len(list_b)
        ratio = 2 #list_b/list_a ratio, depending on agents number and camera number per agent
        if len(rots_list) != m or len(fov_list) != m or n*ratio != m: #"All input lists must have the same length."
            return list_a
        enhanced_a_list = []
        for i in range(n):
            a = list_a[i]
            for j in range(ratio):
                ii = int(i*ratio + j)
                b = list_b[ii]
                fov_radians = fov_list[ii]
                rots = rots_list[ii]
                trans = trans_list[ii]
                affine = affine_matrix[i][j][0]
                C1, h1, w1 = a.shape
                C2, h2, w2 = b.shape
                # Extract trapezoidal and convert to rectangular with width w2
                a_rect = extract_canvas(a, fov_radians, h1, w2, rots, affine)
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
                a_enhanced = restore_canvas(a_enhanced, a.shape, fov_radians, rots, affine)
                a = a + a_enhanced
            enhanced_a_list.append(a)
        return torch.stack(enhanced_a_list)
