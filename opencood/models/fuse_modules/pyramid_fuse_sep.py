# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple, warp_affine_expand
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

def weighted_fuse_debug(x, score, record_len, affine_matrix, align_corners, idx):
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        """
        if H == 256:
            #print("transformation matrix in fuse")
            #print(t_matrix[:, :, :, :])
            t_matrix[i, 1, :, :2] = -t_matrix[i, 1, :, :2]
            t_matrix[i, 1, 1, 2] = -t_matrix[i, 1, 1, 2] #was t_matrix[i, 1, 1, 2]
            #print(t_matrix[:, :, :, :])
        """
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)

        if idx%40 == 0 and H == 128:
            gray_image_function(batch_node_features[b][1], "debug_feature_map/" + str(idx) + str(H) + "outer" )
            gray_image_function(feature_in_ego[1], "debug_feature_map/" + str(idx) + str(H) + "transformed_outer_simple")
            gray_image_function(batch_node_features[b][0], "debug_feature_map/" + str(idx) + str(H) + "ego")
            gray_image_function(feature_in_ego[0], "debug_feature_map/" + str(idx) + str(H) + "transformed_ego_simple")

        """
            expanded_feature_in_ego = warp_affine_expand(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W), align_corners=align_corners)

            gray_image_function(expanded_feature_in_ego[1], "debug_feature_map/" + str(idx) + str(H) + "transformed_outer_expand")
            gray_image_function(expanded_feature_in_ego[0], "debug_feature_map/" + str(idx) + str(H) + "transformed_ego_expand")
        """
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego),
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device),
                                    scores_in_ego)
        #gray_image_function(scores_in_ego[0], "debug_feature_map/" + str(idx) + str(H) + "score0" )
        #gray_image_function(scores_in_ego[1], "debug_feature_map/" + str(idx) + str(H) + "score1" )
        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    if idx%40 == 0 and H == 128:
        gray_image_function(out[0], "debug_feature_map/" + str(idx) + str(H) + "output")
    return out

def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)

    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)

    record_len : list
        shape: (B)

    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3)
    """
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        #print("transformation_matrix would be changed")
        #print(t_matrix[i, :, :, :])
        """
        if W == 256: #CHANGE the transformation matrix to correct version for 1st fusion, and remains correct for further
            print("reversed matrix")
            t_matrix[i, 1, :, :2] = -t_matrix[i, 1, :, :2]
            t_matrix[i, 1, 1, 2] = -t_matrix[i, 1, 1, 2]
        """
        #print(t_matrix[i, :, :, :])
        #gray_image_function(batch_node_features[b][0], "debug_feature_map/" + str(H) + "ego")
        #gray_image_function(batch_node_features[b][1], "debug_feature_map/" + str(H) + "outer")
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        #gray_image_function(feature_in_ego[0], "debug_feature_map/" + str(H) + "transformed_ego")
        #gray_image_function(feature_in_ego[1], "debug_feature_map/" + str(H) + "transformed_outer")
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego),
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device),
                                    scores_in_ego)
        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    #gray_image_function(out[0], "debug_feature_map/" + str(H) + "output")
    return out

class PyramidFusionSep(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck,
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)

        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_ego{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )
            setattr(
                self,
                f"single_head_outer{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_ego{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list

    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list]
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}

        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = None #occ_map = eval(f"self.single_head_ego{i}")(feature_list[i])  # [N, 1, H, W]
            if len(feature_list[i]) == 2:
                occ_map_ego = eval(f"self.single_head_ego{i}")(feature_list[i][0])  # [1, 1, H, W]
                occ_map_outer = eval(f"self.single_head_outer{i}")(feature_list[i][1])
                occ_map = torch.stack([occ_map_ego, occ_map_outer])
                #print("Seperate occ_map and corresponding scores")
            else:
                occ_map = eval(f"self.single_head_ego{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        return fused_feature, occ_map_list

    def forward_collab_debug(self, spatial_features, record_len, affine_matrix, idx, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list]
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}

        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = None #occ_map = eval(f"self.single_head_ego{i}")(feature_list[i])  # [N, 1, H, W]
            if len(feature_list[i]) == 2:
                occ_map_ego = eval(f"self.single_head_ego{i}")(feature_list[i][0])  # [1, 1, H, W]
                occ_map_outer = eval(f"self.single_head_outer{i}")(feature_list[i][1])
                occ_map = torch.stack([occ_map_ego, occ_map_outer])
                #print("Seperate occ_map and corresponding scores")
            else:
                occ_map = eval(f"self.single_head_ego{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask
            fused_feature_list.append(weighted_fuse_debug(feature_list[i], score, record_len, affine_matrix, self.align_corners, idx))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        #print(fused_feature.size())
        if idx%40 == 0:
            gray_image_function(fused_feature[0], "debug_feature_map/" + str(idx) + "fused_feature" )
        return fused_feature, occ_map_list
