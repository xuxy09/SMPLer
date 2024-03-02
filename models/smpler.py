import torch
import torch.nn as nn
import torch.nn.functional as F

import config as project_cfg
from utils.geometry import rot6d_to_rotmat

from models.SMPL_predictor import SMPLPredictor
from models.SMPL_handler import SMPLHandler
from models.baseline import build_hrnet

from models.transformer_basics import (TranformerConfig, BertLayerNorm, TransformerLayer_v1, LayerNormChannel,
                                       BilateralTransformerLayer, GlobalTransformerLayer, sampling_features)


class TransformerUnit_SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.self_trans_layer = TransformerLayer_v1(cfg)
        
    def forward(self, query):
        return self.self_trans_layer(query)


class TransformerUnit_MultiScale_Global(nn.Module):
    """ multi-scale global Transformer
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_scales = 4
        self.trans_layer_list = nn.ModuleList([GlobalTransformerLayer(cfg) for _ in range(self.num_scales)])
    
    def forward(self, query, feat_list, global_score=None):
        """ query: [b, k+2, c]
            feat_list: [b, c, h_i, w_i]
        """
        output_list = []
        for i in range(self.num_scales):
            tmp_global_score = None if global_score is None else global_score[i]
            tmp = self.trans_layer_list[i](query, feat_list[i], spatial_score=tmp_global_score)
            output_list.append(tmp)
        
        output = torch.stack(output_list, dim=-1)
        output = torch.mean(output, dim=-1)    # use mean for aggregation; could try linear+sum or cat+linear
        return output


class TransformerUnit_MultiScale_Local(nn.Module):
    """ multi-scale local Transformer with local positional encoding
    """ 
    def __init__(self, cfg):
        super().__init__()
        r = cfg.sampling_radius
        num_heads = cfg.num_attention_heads
        self.num_local_scales = 1

        self.relative_position_encoding_table = nn.ParameterList([nn.Parameter(torch.zeros(1, num_heads, 2*r+1, 2*r+1)) for _ in range(self.num_local_scales)])
        
        self.trans_layer_list = nn.ModuleList([BilateralTransformerLayer(cfg) for _ in range(self.num_local_scales)])
    
    def forward(self, query, local_feat_list, local_spat_list):
        """ query: [b, k, c]
            local_feat_list: {[b, c, k, (2r)^2]}
            local_spat_list: {[b, k, (2r)^2, 2]}
        """
        output_list = []
        for i in range(self.num_local_scales):
            sampled_feat = local_feat_list[i]
            spatial_dist = local_spat_list[i]

            b, c, k, n = sampled_feat.shape
            spatial_score = F.grid_sample(self.relative_position_encoding_table[i].expand(b, -1, -1, -1), 
                                        spatial_dist, mode='bilinear', align_corners=True)   # [b, num_heads, k, n]
            spatial_score = spatial_score.permute(0, 2, 1, 3).reshape(b*k, -1, 1, n)   # [b*k, num_heads, 1, n]
            
            output_list.append(self.trans_layer_list[i](query, sampled_feat, spatial_score))
        
        output = torch.stack(output_list, dim=-1)
        output = torch.mean(output, dim=-1)
        
        return output   # [b, k, c]


class TransformerUnit_Assembled(nn.Module):
    """ composed of: 
            1) global Transformer layer
            2) local Transformer layer
            3) self-attention layer
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.global_unit = TransformerUnit_MultiScale_Global(cfg)
        
        self.local_unit = TransformerUnit_MultiScale_Local(cfg)
        
        self.self_trans_layer = TransformerUnit_SelfAttention(cfg)
    
    def forward(self, query, feat_list, local_feat_list, local_spat_list):
        """ query: [b, k+2, c]
            feat_list: [b, c, h_i, w_i]
        """
        global_query = self.global_unit(query, feat_list)
        local_query = self.local_unit(query[:, :-2], local_feat_list, local_spat_list)
        global_query[:, :-2] = 0.5 * (global_query[:, :-2] + local_query)
        return self.self_trans_layer(global_query)


# ****** Define Transformer Block ******
class TransformerBlock(nn.Module):
    """ Transformer Block for SMPLer
        no local attention
    """
    def __init__(self, config, pos_embeddings=None):
        super().__init__()
        self.config = config
        self.scales = [0, 1, 2, 3] #self.config.global_compute_scales

        # query position encoding
        self.early_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) if pos_embeddings is None else pos_embeddings
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # feature position encoding
        feat_res_list = config.feat_res_list
        self.feat_position_embedding = nn.Parameter(torch.randn(1, config.hidden_size, feat_res_list[0], feat_res_list[0]) * 0.1)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.trans_units = nn.ModuleList([TransformerUnit_Assembled(config) for _ in range(config.num_units_per_block)])
        
        # output
        self.head = nn.Linear(config.hidden_size, config.hidden_size)
        self.residual = nn.Linear(config.hidden_size, config.hidden_size)

        self.apply(self.init_weights)
    
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, query, feat_list, local_feat_list, local_spat_list):
        """ query: [b, k+2, c]
            feat_list: {[b, c, h_i, w_i]}, i=0,1,2,3
            local_feat_list: {[b, c, k, (2r)^2]}
            local_spat_list: {[b, k, (2r)^2, 2]}
        """
        # query position embedding
        b, seq_length, c = query.shape
        query_embed = self.early_embedding(query)   # linear

        position_ids = torch.arange(seq_length, dtype=torch.long, device=query.device)
        position_ids = position_ids.unsqueeze(0).expand(b, -1)
        position_embeddings = self.position_embeddings(position_ids)
        query_embed = position_embeddings + query_embed
        query_embed = self.dropout(query_embed)    # dropout

        # feature position embedding
        new_feat_list = []
        for i in range(self.scales[-1]+1):
            if i == 0:
                feat_pos_embed = self.feat_position_embedding
            else:
                feat_pos_embed = self.pooling(feat_pos_embed)
            
            if i in self.scales:
                new_feat_list.append(feat_list[i] + feat_pos_embed)
        
        # apply Transformer Units
        for unit in self.trans_units:
            query_embed = unit(query_embed, new_feat_list, local_feat_list, local_spat_list)
        
        # output
        output = self.head(query_embed) + self.residual(query)
        return output


# ****** Define Transformer Block Wrapper ******
class TransformerBlock_Wrapped(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sampling_radius = cfg.sampling_radius

        # image feature projection
        self.scales = [0, 1, 2, 3] 
        self.local_scales = [0] 
        self.all_scales = sorted(list(set(self.scales + self.local_scales)))

        raw_feat_dim = cfg.raw_feat_dim
        self.raw_feat_projection_layer = nn.ModuleDict({str(s): nn.Sequential(nn.Conv2d(raw_feat_dim[s], cfg.hidden_size, 1, 1), LayerNormChannel(cfg.hidden_size, cfg.layer_norm_eps)) 
                                                        for s in self.all_scales})

        # joint feature projection for local attention
        self.joint_feat_projection_layer = nn.ModuleDict({str(s): nn.Linear(cfg.hidden_size, cfg.hidden_size) for s in self.local_scales})

        # Transformer
        self.trans_block = TransformerBlock(cfg)
        
        # predictors
        self.theta_predictor = nn.Linear(cfg.hidden_size, 6)
        self.beta_predictor = nn.Linear(cfg.hidden_size, 10)
        self.cam_predictor = nn.Linear(cfg.hidden_size, 3)

        self.init_predictors()
    
    @torch.no_grad()
    def init_predictors(self):
        self.beta_predictor.weight.data.normal_(0, 1e-3)
        self.beta_predictor.bias.data.zero_()
        self.cam_predictor.weight.data.normal_(0, 1e-3)
        self.cam_predictor.bias.data.zero_()

        self.theta_predictor.weight.data.normal_(0, 1e-3)
        self.theta_predictor.bias.data.zero_()
        self.theta_predictor.bias.data[[0,3]] = 0.1
    
    def compute_spatial_dist(self, sample_loc, joints_2d, h, w, radius):
        """ @ inputs:
                - sample_loc: [b, k, (2r)^2, 2]
                - joints_2d: [b, k, 2]
            @ outputs:
                - output: [b, k, (2r)^2, 2]
        """
        sample_loc_x, sample_loc_y = torch.split(sample_loc, 1, dim=-1)
        sample_loc_x, sample_loc_y = sample_loc_x.squeeze(-1), sample_loc_y.squeeze(-1)
        joints_2d_x, joints_2d_y = torch.split(joints_2d, 1, dim=-1)

        spatial_dist_x = (sample_loc_x - joints_2d_x) / 2 * (w-1) / radius
        spatial_dist_y = (sample_loc_y - joints_2d_y) / 2 * (h-1) / radius

        return torch.stack([spatial_dist_x, spatial_dist_y], dim=-1)

    def forward(self, query, feat_list, joints_2d=None):
        """ @ inputs: 
                - query: [b, k+2, c]
                - feat_list: {[b, c, h_i, w_i]}, i=0,1,2,3
                - joints_2d: [b, k, 2]

            @ outputs:
                - rotmat: [b, k, 3, 3]
                - beta: [b, 10]
                - cam: [b, 3]
                - query_new: [b, k+2, c]
        """
        # image feature projection
        new_feat_list = []
        for i in range(len(feat_list)):
            if i in self.all_scales:
                new_feat_list.append(self.raw_feat_projection_layer[str(i)](feat_list[i]))
            else:
                new_feat_list.append(None)

        # local feature sampling
        local_feat_list = []
        local_spat_list = []    # sampling locations for local positional encoding
        joint_feat_projected = []
        for i in self.local_scales:
            feat = new_feat_list[i]
            sampled_feat, spatial_loc, joint_feat = sampling_features(feat, joints_2d, self.sampling_radius)   # [b, c, k, (2r)^2], [b, k, (2r)^2, 2], [b, c, k]
            spatial_dist = self.compute_spatial_dist(spatial_loc, joints_2d, feat.shape[-2], feat.shape[-1], self.sampling_radius)    # [b, k, (2r)^2, 2]
            local_feat_list.append(sampled_feat)
            local_spat_list.append(spatial_dist)

            joint_feat_projected.append(self.joint_feat_projection_layer[str(i)](joint_feat.transpose(1, 2)))
        
        if len(joint_feat_projected) > 0:
            joint_feat_projected = torch.stack(joint_feat_projected, dim=-1).mean(-1)   # [b, k, c]
            query_1 = query[:, :-2] + joint_feat_projected
            query_2 = query[:, -2:] + torch.mean(joint_feat_projected, dim=1, keepdim=True)
            query = torch.cat([query_1, query_2], dim=1)

        # Transformer
        query_new = self.trans_block(query, new_feat_list, local_feat_list, local_spat_list)
        
        # predictors
        theta = self.theta_predictor(query_new[:, :-2])
        rotmat = rot6d_to_rotmat(theta).reshape(theta.shape[0], -1, 3, 3)
        beta = self.beta_predictor(query_new[:, -2])
        cam = self.cam_predictor(query_new[:, -1])
        return rotmat, beta, cam, query_new


class SMPLer(nn.Module):
    """ Predict SMPL
    """
    def __init__(self, args, trans_cfg=None):
        super().__init__()

        # *** Config for Transformer; overriden by args ***
        if trans_cfg is None:
            trans_cfg = TranformerConfig()
            
        # *** baseline for feature extraction and initial estimation ***
        self.backbone = build_hrnet(args.hrnet_type)
        self.SMPL_predictor = SMPLPredictor(args=args)

        if args.data_mode == 'h36m':
            self.smpl_handler = SMPLHandler(path_to_regressor=project_cfg.JOINT_REGRESSOR_H36M_correct)
        else:
            self.smpl_handler = SMPLHandler(path_to_regressor=project_cfg.JOINT_REGRESSOR_3DPW)
        self.backbone_feat_dims = project_cfg.hrnet_dict[args.hrnet_type][2]

        # *** Linear layers for query initialization ***
        hidden_size = trans_cfg.hidden_size 

        self.init_query_layer_beta = nn.Linear(10, hidden_size)
        self.init_query_layer_theta = nn.Linear(3*3, hidden_size)
        self.init_query_layer_cam = nn.Linear(3, hidden_size)
        self.global_feat_projection = nn.Linear(self.backbone_feat_dims[-1], hidden_size)
        # ***********************************************

        # *** Transformer Blocks ***
        self.num_transformers = args.num_transformers
        transformer_list = []
        for i in range(self.num_transformers):
            transformer_list.append(TransformerBlock_Wrapped(trans_cfg))
        self.transformer = nn.ModuleList(transformer_list)
        # ***************************

    def compute_init_query(self, pred_rotmat, pred_shape, pred_cam, global_feat):
        query_beta = self.init_query_layer_beta(pred_shape)
        query_cam = self.init_query_layer_cam(pred_cam)
        query_theta = self.init_query_layer_theta(pred_rotmat.flatten(2))

        query = torch.cat([query_theta, query_beta.unsqueeze(1), query_cam.unsqueeze(1)], dim=1)    # [b, 24+2, c]
        
        global_feat = self.global_feat_projection(global_feat)
        return query + global_feat.unsqueeze(1)

    def forward(self, img):
        """
        # img --> (backbone) --> feat_list --> global_feat
        # globa_feat --> (SMPL_predictor) --> theta, beta, cam
        # theta, beta, cam --> (SMPL_handler) --> joints_2d

        # theta, beta, cam, global_feat --> (compute_init_query) --> query
        # query, feat_list, joints_2d --> (Transformer) --> new_theta, new_beta, new_cam, new_query
        # new_theta, new_beta, new_cam --> (SMPL_handler) --> joints_2d
        """
        # baseline
        feat_list = self.backbone(img) 
        global_feat = feat_list[-1].flatten(2).mean(2)
        pred_rotmat, pred_shape, pred_cam = self.SMPL_predictor(global_feat, global_pooling=False) 
        pred_smpl_dict = self.smpl_handler(pred_rotmat, pred_shape, theta_form='rot-matrix', cam=pred_cam)

        # initialize query for Transformer
        query = self.compute_init_query(pred_rotmat, pred_shape, pred_cam, global_feat)

        # --- Run all Transformers ---
        pred_smpl_dicts = [pred_smpl_dict]
        for i in range(self.num_transformers):
            delta_theta, delta_beta, delta_cam, query = self.transformer[i](query, feat_list, pred_smpl_dict['smpl_joints2d'])

            pred_shape = pred_shape + delta_beta
            pred_cam = pred_cam + delta_cam
            pred_rotmat = torch.matmul(pred_rotmat, delta_theta)
            pred_smpl_dict = self.smpl_handler(pred_rotmat, pred_shape, theta_form='rot-matrix', cam=pred_cam)
            pred_smpl_dicts.append(pred_smpl_dict)
        # --- End of Transformers ---

        return pred_smpl_dicts


    
    
    
