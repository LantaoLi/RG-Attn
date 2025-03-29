# -*- coding: utf-8 -*-
# Author: Lantao Li <lantao.li@sony.com>
# License: TDG-Attribution-NonCommercial-NoDistribution

"""
RG-crossattention-Paint2Puzzel-DairV2X
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.fuse_modules.cross_fuse import DepthAwareCrossAttention
from opencood.models.fuse_modules.co_coloring import DepthCocoloring
from opencood.utils.transformation_utils import normalize_pairwise_tfm, normalize_pairwise_tfm_2duplicate
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
import importlib
import torchvision

from PIL import Image
import torchvision.transforms.functional as TF
import datetime
import copy
import time

class HeterPyramidCollabAllcross(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollabAllcross, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {}

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion for cross-agent;
        cross-modality will be set according to config.yaml (PTP default);
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.rg_arch = args['cross_fusion']['arch']
        if self.rg_arch == 'CoSCoCo':
            self.cross_fusion = DepthCocoloring(args['cross_fusion']).cuda()
        else:
            self.cross_fusion = DepthAwareCrossAttention(args['cross_fusion']).cuda()
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def gray_image_function(self, input_tensor, image_info):
        gray_tensor = input_tensor.sum(dim=0, keepdim=True)
        gray_tensor = gray_tensor.mul(255).byte()
        gray_image = TF.to_pil_image(gray_tensor)
        now = datetime.datetime.now()
        filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
        gray_image.save(filename)

    def gray_occ_function(self, input_tensor, image_info):
        input_tensor = input_tensor.squeeze(0)
        tensor_normalized = (input_tensor + 10)/20
        gray_image = TF.to_pil_image(tensor_normalized)
        now = datetime.datetime.now()
        filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
        gray_image.save(filename)

    def occ_projection(self, occ_map, record_len, affine_matrix):
        if record_len <= 1:
            return score
        score = occ_map
        B, L = affine_matrix.shape[:2]
        split_score = regroup(score, record_len)
        scores_in_ego_list = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            #print(N)
            score = split_score[b]
            t_matrix = affine_matrix[b][:N, :N, :, :]
            i = 0 # ego
            scores_in_ego = warp_affine_simple(split_score[b],
                                               t_matrix[i, :, :, :],
                                               (128, 256), align_corners=False)
            scores_in_ego_list.append(scores_in_ego)
        scores_in_ego_list = torch.stack(scores_in_ego_list)
        return scores_in_ego_list

    def score_projection(self, occ_map, record_len, affine_matrix):
        if record_len <= 1:
            return score
        score = torch.sigmoid(occ_map) + 1e-4
        B, L = affine_matrix.shape[:2]
        split_score = regroup(score, record_len)
        scores_in_ego_list = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            score = split_score[b]
            t_matrix = affine_matrix[b][:N, :N, :, :]
            i = 0 # ego
            scores_in_ego = warp_affine_simple(split_score[b],
                                               t_matrix[i, :, :, :],
                                               (128, 256), align_corners=False)
            scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
            scores_in_ego = torch.softmax(scores_in_ego, dim=0)
            scores_in_ego = torch.where(torch.isnan(scores_in_ego),
                                        torch.zeros_like(scores_in_ego, device=scores_in_ego.device),
                                        scores_in_ego)
            scores_in_ego_list.append(scores_in_ego)
        scores_in_ego_list = torch.stack(scores_in_ego_list)
        return scores_in_ego_list

    def derive_mask(self, mask, threshold=-5.0):
        """
        this function is used for deriving a BEV space mask slightly larger than original one
        be very careful with the threshold value!
        """
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        valid_points = (mask > threshold).float()
        expanded_mask = F.conv2d(valid_points.unsqueeze(0), kernel, padding=1)
        new_mask = (expanded_mask > 0).float()
        return new_mask.squeeze(0)

    def and_merge_mask(self, mask2, mask3):
        """
        this function is used for deriving a BEV space mask merge (and valid)
        """
        return mask2*mask3

    def gray_score_function(self, input_tensor, image_info):
        input_tensor = input_tensor.squeeze(0)
        tensor_normalized = (input_tensor)
        gray_image = TF.to_pil_image(tensor_normalized)
        now = datetime.datetime.now()
        filename = f"{image_info}_gray_image_{now.strftime('%m%d%H%M%S')}.png"
        gray_image.save(filename)

    def color_image_function(self, input_tensor, image_info): #for 256 channels
        C, H, W = input_tensor.shape
        device = input_tensor.device
        tensor_1 = input_tensor[:86, :, :].sum(dim=0)/20.0
        tensor_2 = input_tensor[86:171, :, :].sum(dim=0)/20.0
        tensor_3 = input_tensor[171:256, :, :].sum(dim=0)/20.0
        color_mapped_tensor = torch.stack([tensor_1, tensor_2, tensor_3], dim=0)
        color_mapped_tensor = color_mapped_tensor.mul(255).byte()
        color_image = TF.to_pil_image(color_mapped_tensor)
        now = datetime.datetime.now()
        filename = f"{image_info}_color_image_{now.strftime('%m%d%H%M%S')}.png"
        color_image.save(filename)

    def cam_fov(self, data_dict, modality_name):
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        cam_K = image_inputs_dict['intrins']
        f_x = cam_K[:, :, 0, 0]
        c_x = cam_K[:, :, 0, 2]
        image_width, image_height = 1920.0, 1080.0 #settings for dair-v2x
        fov_x = 2*torch.arctan(image_width/(2.0*f_x))
        return fov_x, f_x, c_x

    def cam_rots_trans(self, data_dict, modality_name):
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        rots = image_inputs_dict['rots']
        trans = image_inputs_dict['trans']
        return rots, trans

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list']
        print(agent_modality_list)
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        fov, fxs, cxs = self.cam_fov(data_dict, 'm2')
        rots, trans = self.cam_rots_trans(data_dict, 'm2')

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            if modality_name == 'm1':
                feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
                feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        lidar_feature_2d_list = []
        lidar_setting_list = []
        camera_feature_2d_list = []
        camera_setting_list = []
        for modality_name in self.modality_name_list: # was in agent_modality_list
            if len(modality_feature_dict[modality_name]) == 1:
                if modality_name == 'm1':
                    lidar_feature_2d_list.append(modality_feature_dict[modality_name])
                if modality_name == 'm2':
                    camera_feature_2d_list.append(modality_feature_dict[modality_name])
                print(str(modality_name))
            elif len(modality_feature_dict[modality_name]) > 1:
                for agent_i in range(len(modality_feature_dict[modality_name])):
                    if modality_name == 'm1':
                        lidar_feature_2d_list.append(modality_feature_dict[modality_name][agent_i])
                    if modality_name == 'm2':
                        camera_feature_2d_list.append(modality_feature_dict[modality_name][agent_i])
                    print(str(modality_name)+str(agent_i))
        """
        merging lidar and camera first for trial here!!!!
        depending on which specific fusion module to use, the forward function should take different inputs
        """
        if self.rg_arch == 'CoSCoCo':
            print('CoSCoCo')
            heter_feature_2d = torch.stack(lidar_feature_2d_list)
            if self.compress:
                heter_feature_2d = self.compressor(heter_feature_2d)
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    heter_feature_2d,
                                                    record_len,
                                                    affine_matrix,
                                                    agent_modality_list,
                                                    None # previous was self.cam_crop_info, now we make it None to avoid process of camera modality anymore
                                                )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
            fused_feature = self.cross_fusion.forward(fused_feature, camera_feature_2d_list, affine_matrix, fov, rots, trans)
        elif self.rg_arch == 'PTP':
            print('PTP')
            heter_feature_2d_list = self.cross_fusion.forward(lidar_feature_2d_list, camera_feature_2d_list, fov, rots, trans)
            heter_feature_2d = torch.stack(heter_feature_2d_list)
            if self.compress:
                heter_feature_2d = self.compressor(heter_feature_2d)
            # heter_feature_2d is downsampled 2x
            # add croping information to collaboration module
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                    heter_feature_2d,
                                                    record_len,
                                                    affine_matrix,
                                                    agent_modality_list,
                                                    None # previous was self.cam_crop_info, now we make it None to avoid process of camera modality anymore
                                                )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        output_dict.update({'occ_single_list':
                            occ_outputs})

        return output_dict
