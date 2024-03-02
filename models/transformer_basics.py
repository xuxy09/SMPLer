import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TranformerConfig:
    def __init__(self):
        self.raw_feat_dim = [32, 64, 128, 256]
        self.feat_res_list = [56, 28, 14, 7]
        self.sampling_radius = 4

        self.hidden_size = 128
        self.interm_size_scale = 2
        self.num_attention_heads = 4
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.initializer_range = 0.02
        self.max_position_embeddings = 24 + 2
        self.num_units_per_block = 2


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def attention_dotproduct(self, query_layer, key_layer, value_layer):
        """ Low-Rank Attention from '3D Human Texture Estimation from a Single Image with Transformers', ICCV 2021
        """
        attention_scores = torch.matmul(key_layer.transpose(-1, -2), value_layer) / value_layer.shape[-2]
        context_layer = torch.matmul(query_layer, attention_scores) / self.attention_head_size
        return context_layer 

    def forward(self, hidden_states, key_states, value_states, spatial_score=None, attention_type=0):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if attention_type == 0:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # Normalize the attention scores to probabilities.
            if spatial_score is None:
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
            else:
                attention_probs = nn.Softmax(dim=-1)(attention_scores + spatial_score)   # [b, num_heads, n_q, n_k]

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
        elif attention_type == 1:
            context_layer = self.attention_dotproduct(query_layer, key_layer, value_layer)
        else:
            raise ValueError('Unknow attention type{}'.format(attention_type))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer    # no need to output "attention_probs"
    
    @staticmethod
    def my_ops(m, x, y):
        """ m: the module itself
            x: module input
            y: module output
        """
        q = x[0]
        k = x[1]
        # print('here', q.shape, k.shape)
        matmul_ops = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[1] 
        m.total_ops += torch.DoubleTensor([int(matmul_ops)])


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
    @staticmethod
    def my_ops(m, x, y):
        nelements = x[0].numel()
        m.total_ops += torch.DoubleTensor([int(nelements)])

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x
    
    @staticmethod
    def my_ops(m, x, y):
        nelements = x[0].numel()
        m.total_ops += torch.DoubleTensor([int(nelements)])


class TransformerLayer_v1(nn.Module):
    """ Early LayerNorm:
        input -> (Norm) -> (Attention)  -> (Linear) -> (Drop) -> (Add) ->
              -> (Norm) -> (Linear) -> (Gelu) -> (Linear) -> (Drop) -> (Add) -> output
    """
    def __init__(self, config):
        super().__init__()
        self.LayerNorm_attn = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = BertSelfAttention(config)  # attention layer
        self.linear_attn = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        LayerNorm_mlp = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        mlp_linear1 = nn.Linear(config.hidden_size, config.hidden_size * config.interm_size_scale)
        mlp_linear2 = nn.Linear(config.hidden_size * config.interm_size_scale, config.hidden_size)
        intermediate_act_fn = nn.GELU()
        self.mlp = nn.Sequential(LayerNorm_mlp, mlp_linear1, intermediate_act_fn, mlp_linear2, nn.Dropout(config.hidden_dropout_prob))   # changed to use a new dropout just for clarity

    def forward(self, hidden_states, key_states=None, value_states=None, spatial_score=None):
        hidden_states_norm = self.LayerNorm_attn(hidden_states)
        if key_states is None:   # self-attention
            key_states = hidden_states_norm
            value_states = hidden_states_norm
        elif value_states is None:   # cross-attention
            value_states = key_states
        
        attention_output = self.attention(hidden_states_norm, key_states, value_states, spatial_score)
        attention_output = self.linear_attn(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + hidden_states

        mlp_output = self.mlp(attention_output)
        return mlp_output + attention_output


class GlobalTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer_layer = TransformerLayer_v1(config)

    def forward(self, query, feat, spatial_score=None):
        """ query: [b, k, c]
            feat: [b, c, h, w]
        """
        feat = feat.flatten(-2).transpose(1, 2)  # [b, h*w, c]
        output = self.transformer_layer(query, feat, spatial_score=spatial_score)   # cross attention
        return output


class BilateralTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer_layer = TransformerLayer_v1(config)

    def forward(self, query, feat, spatial_score=None):
        """ query: [b, k, c]
            feat: [b, c, k, n], ie, batch_size, channels, num_keypoints, num_samples_per_keypoint
            spatial_score: [b*k, 1, 1, n] or [b*k, num_heads, 1, n]
        """
        b, c, k, n = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(b*k, n, c)  # [b*k, n, c]
        query = query.reshape(b*k, 1, c)   # [b*k, 1, c]
    
        # cross attention
        output = self.transformer_layer(query, feat, spatial_score=spatial_score)   # [b*k, 1, c]
        return output.view(b, k, c)


def sampling_features(feat, smpl_joints2d, radius):
    """ feat: [b, c, h, w]
        smpl_joints2d: [b, k, 2]
    """
    joint_feat = F.grid_sample(feat, smpl_joints2d.unsqueeze(2), mode='bilinear', align_corners=True).squeeze(-1)  # [b, c, k]

    b, c, h, w = feat.shape
    smpl_joints2d_x = (smpl_joints2d[..., 0] + 1) * 0.5 * (w-1)  # [b, k]
    smpl_joints2d_y = (smpl_joints2d[..., 1] + 1) * 0.5 * (h-1)
    radius_vec = torch.arange(-(radius-1), radius+1, device=feat.device)  # [2*r]

    sample_x_all = smpl_joints2d_x.floor().unsqueeze(-1) + radius_vec.unsqueeze(0).unsqueeze(0)  # [b, k, 2*r]
    sample_y_all = smpl_joints2d_y.floor().unsqueeze(-1) + radius_vec.unsqueeze(0).unsqueeze(0)
    
    sample_x_all = (sample_x_all / (w-1)).clamp(0, 1) * 2 - 1
    sample_x_all = sample_x_all.unsqueeze(-1).expand(-1, -1, -1, 2*radius)   # [b, k, 2*r, 2*r]

    sample_y_all = (sample_y_all / (h-1)).clamp(0, 1) * 2 - 1
    sample_y_all = sample_y_all.unsqueeze(-2).expand(-1, -1, 2*radius, -1)   

    sample_coord_all = torch.stack([sample_x_all, sample_y_all], dim=-1)   # [b, k, 2*r, 2*r, 2]
    sample_coord_all = sample_coord_all.view(b, -1, (2*radius)**2, 2)   # [b, k, (2*r)^2, 2]

    sampled_feat = F.grid_sample(feat, sample_coord_all, mode='nearest', align_corners=True)   # [b, c, k, (2*r)^2]
    return sampled_feat, sample_coord_all, joint_feat



