import torch
import torch.nn as nn
import numbers
from typing import List
import math
import numpy as np
from torch.nn import functional as F

def daf_weight_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    total_loss = 0
    batch, channel_num, imh, imw = edges.shape
    for b_i in range(batch):
        p = preds[b_i, :, :, :].unsqueeze(1)
        t = edges[b_i, :, :, :].unsqueeze(1)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight[t > 0.5] = num_neg / (num_pos + num_neg)
        weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # Calculate loss.
        loss = torch.nn.functional.binary_cross_entropy(p.float(), t.float(), weight=weight, reduction='none')
        loss = torch.sum(loss)
        total_loss = total_loss + loss
    return total_loss/batch#标准正负例loss，batch sum, img sum#对batch平均

def daf_loss(preds,edges,weight=[0.5,0.5,0.5,0.5,0.5,1.1]):
    loss=0
    for i in range (len(preds)):
        loss+=daf_weight_loss(preds[i],edges)*weight[i]
    return loss

class VGG16_C(nn.Module):
    """"""
    def __init__(self, pretrain=None, logger=None):
        super(VGG16_C, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)
        if pretrain:
            if '.npy' in pretrain:
                state_dict = np.load(pretrain).item()
                for k in state_dict:
                    state_dict[k] = torch.from_numpy(state_dict[k])
            else:
                state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    if logger:
                        logger.info('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 0.01)
        else:
            self._initialize_weights(logger)

    def forward(self, x):
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.relu2_1(self.conv2_1(pool1))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.relu3_1(self.conv3_1(pool2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        pool3 = self.pool3(conv3_3)
        conv4_1 = self.relu4_1(self.conv4_1(pool3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_3)
        # pool4 = conv4_3
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))

        side = [conv1_1, conv1_2, conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3, conv4_1,
                conv4_2, conv4_3, conv5_1, conv5_2, conv5_3]
        return side

    def _initialize_weights(self, logger=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class RelativePosition(nn.Module):
    """
    Relative Position Embeddings Module

    This module generates learnable relative position embeddings to enrich
    the self-attention mechanism with information about the relative distances
    between elements in input sequences.

    Args:
        d_a (int): Number of dimensions in the relative position embeddings.
        k (int): Clipping distance.

    Attributes:
        position_embeddings (nn.Parameter): Learnable parameter for relative position embeddings.

    Example:
        >>> # Create a RelativePosition instance with 16 dimensions and clipping distance of 10
        >>> relative_position = RelativePosition(d_a=16, k=10)
        >>> # Generate relative position embeddings for sequences of lengths 5 and 7
        >>> embeddings = relative_position(length_query=5, length_key=7)
    """

    def __init__(self, d_a: int, k: int):
        """
        Initialize the RelativePosition module.

        Args:
        - d_a (int): Number of dimensions in the relative position embeddings.
        - k (int): Clipping distance.
        """
        super().__init__()
        self.d_a = d_a
        self.k = k
        self.position_embeddings = nn.Parameter(torch.empty((2 * k + 1, d_a)))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, length_query: int, length_key: int) -> torch.Tensor:
        """
        Compute relative position embeddings.

        Args:
        - length_query (int): Length of the query sequence.
        - length_key (int): Length of the key sequence.

        Returns:
        - embeddings (torch.Tensor): Relative position embeddings (length_query, length_key, embedding_dim).
        """
        # Generate relative position embeddings
        indices_query = torch.arange(length_query, device=self.position_embeddings.device)
        indices_key = torch.arange(length_key, device=self.position_embeddings.device)
        distance_matrix = indices_key.unsqueeze(0) - indices_query.unsqueeze(1)
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.k, self.k)
        final_matrix = distance_matrix_clipped + self.k
        embeddings = self.position_embeddings[final_matrix.to(torch.long)]

        return embeddings


class RelationAwareAttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim, k_bias_matrix, v_bias_matrix):
        """
        Initializes the RelationAwareAttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
            k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
            v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.k_bias_matrix = k_bias_matrix
        self.v_bias_matrix = v_bias_matrix

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query) # (b_s, n_t, head_dim)
        key: torch.Tensor = self.key_weights(key) # (b_s, n_t, head_dim)
        value: torch.Tensor = self.value_weights(value) # (b_s, n_t, head_dim)

        # Self-Attention scores
        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) # Q*K^T:(b_s, n_t, n_t)

        # Relative Position Attention scores
        attn_2: torch.Tensor = torch.matmul(query.permute(1, 0, 2), self.k_bias_matrix.transpose(1, 2)).transpose(0, 1) # Q*K_shifting^T:(b_s, n_t, n_t)

        # Relation-aware Self-Attention scores
        att_scores: torch.Tensor = (attn_1 + attn_2)/self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)

        # Weighted sum of values
        values_1: torch.Tensor = torch.matmul(att_weights, value) # (b_s, n_t, head_dim)

        # Relative Position Representation for values
        values_2: torch.Tensor = torch.matmul(att_weights.permute(1, 0, 2), self.v_bias_matrix).transpose(0, 1) # (b_s, n_t, head_dim)

        # Relation-aware values
        n_value  = values_1 + values_2
        return n_value


class RelationAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        k (int): Clipping distance for relative position embeddings.
        seq_len (int): Length of the input sequences.

    Attributes:
        hidden_size (int): Hidden size for the model (embedding dimension).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        relative_position_k (RelativePosition): Instance of RelativePosition for query-key relative positions.
        relative_position_v (RelativePosition): Instance of RelativePosition for query-value relative positions.
        k_bias_matrix (torch.Tensor): Matrix for relative position attention in query-key interaction.
        v_bias_matrix (torch.Tensor): Matrix for relative position attention in query-value interaction.
        attention_heads (nn.ModuleList): List of RelationAwareAttentionHead layers.
        fc (nn.Linear): Fully connected layer for final projection.
    """

    def __init__(self, hidden_size, num_heads, k, seq_len):
        """
        Initializes the RelationAwareMultiHeadAttention layer.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            num_heads (int): Number of attention heads.
            k (int): Clipping distance for relative position embeddings.
            seq_len (int): Length of the input sequences.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.relative_position_k: torch.Tensor = RelativePosition(self.head_dim, k)
        self.relative_position_v: torch.Tensor = RelativePosition(self.head_dim, k)
        self.k_bias_matrix: torch.Tensor = self.relative_position_k(seq_len, seq_len)
        self.v_bias_matrix: torch.Tensor = self.relative_position_v(seq_len, seq_len)
        self.attention_heads: nn.ModuleList = nn.ModuleList([RelationAwareAttentionHead(self.hidden_size, self.head_dim,
                                                                 self.k_bias_matrix, self.v_bias_matrix) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(hidden_size, hidden_size)
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated hidden state after applying multi-head attention mechanism.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias



class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)



    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.norm1(x)
        attn_00 = self.conv0_1(x1)
        attn_01 = self.conv0_2(x1)
        attn_10 = self.conv1_1(x1)
        attn_11 = self.conv1_2(x1)
        attn_20 = self.conv2_1(x1)
        attn_21 = self.conv2_2(x1)
        out1 = attn_00 + attn_10 + attn_20
        out2 = attn_01 + attn_11 + attn_21
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x
        return out


# @MODELS.register_module()
# class MCAHead(BaseDecodeHead):
#     def __init__(self, in_channels, image_size, heads, c1_channels,
#                  **kwargs):
#         super(MCAHead, self).__init__(in_channels, input_transform='multiple_select', **kwargs)
#         self.image_size = image_size
#         self.decoder_level = Attention(in_channels[1], heads, LayerNorm_type='WithBias')
#         self.align = ConvModule(
#             in_channels[3],
#             in_channels[0],
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         self.squeeze = ConvModule(
#             sum((in_channels[1], in_channels[2], in_channels[3])),
#             in_channels[1],
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg)
#         self.sep_bottleneck = nn.Sequential(
#             DepthwiseSeparableConvModule(
#                 in_channels[1] + in_channels[0],
#                 in_channels[3],
#                 3,
#                 padding=1,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg),
#             DepthwiseSeparableConvModule(
#                 in_channels[3],
#                 in_channels[3],
#                 3,
#                 padding=1,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg))
#
#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)
#         inputs = [resize(
#             level,
#             size=self.image_size,
#             mode='bilinear',
#             align_corners=self.align_corners
#         ) for level in inputs]
#         y1 = torch.cat([inputs[1], inputs[2], inputs[3]], dim=1)
#         x = self.squeeze(y1)
#         x = self.decoder_level(x)
#         x = torch.cat([x, inputs[0]], dim=1)
#         x = self.sep_bottleneck(x)
#
#         output = self.align(x)
#         output = self.cls_seg(output)
#         return output


import torch
from torch import nn

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充
        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图



class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 获取输入张量的形状信息
        b, c, h, w = x.shape
        # 计算像素点数量
        n = h * w - 1
        # 计算输入张量在通道维度上的均值
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        # 计算输入张量在通道维度上的方差
        var = torch.sum(torch.pow((x - mean), 2), dim=[2, 3], keepdim=True) / n
        # 计算特征图的激活值
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        # 使用 Sigmoid 函数进行归一化
        out = self.sigmoid(e_t) * x
        return out


class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number. 官方的代码中设为512
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h * w
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        # linear_0是第一个memory unit
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n

        x = self.linear_1(attn)  # b, c, n
        # linear_1是第二个memory unit
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

class EGAM(nn.Module):
    def __init__(self, in_channels):
        super(EGAM, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels , in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.Siam=SimAM()

        # self.cbam = CBAM(in_channels)
        # self.ECA=macnnet.ECAAttention()

    def forward(self, edge_feature, x):
        residual = x
        xsize = x.size()[2:]
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input
        # print(input_feature.shape)
        # fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(input_feature)
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map
        out = 0.5*fusion_feature + residual
        # out = self.ECA(out)
        out=self.Siam(out)
        return out








# import torch
# from torch import nn


# 定义ECA注意力模块的类
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法


class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )




    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y




import torch
import torch.nn as nn
import math

def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k
class TFAM(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse
# 导入PyTorch相关模块，用于构建和训练神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个函数来生成位置编码，返回一个包含位置信息的张量
def position(H, W, is_cuda=True):
    # 生成宽度和高度的位置信息，范围在-1到1之间
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)# 为宽度生成线性间距的位置信息并复制到GPU
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W) # 为高度生成线性间距的位置信息并复制到GPU
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1) # 在CPU上为宽度生成线性间距的位置信息
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W) # 在CPU上为高度生成线性间距的位置信息
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0) # 合并宽度和高度的位置信息，并增加一个维度
    return loc

# 定义一个函数实现步长操作，用于降采样
def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride] # 通过步长来降低采样率

# 初始化函数，将张量的值填充为0.5
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5) # 使用0.5来填充张量

# 初始化函数，将张量的值填充为0
def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

# 定义ACmix模块的类
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__() # 调用父类的构造函数
        # 初始化模块参数
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))  # 注意力分支权重
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))  # 卷积分支权重
        self.head_dim = self.out_planes // self.head  # 每个头的维度

        # 定义用于特征变换的卷积层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)  # 位置编码的卷积层

        # 定义自注意力所需的padding和展开操作
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        # 定义用于生成动态卷积核的全连接层和深度可分离卷积层
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)# 深度可分离卷积层，用于应用动态卷积核

        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        init_rate_half(self.rate1)  # 初始化注意力分支权重为0.5
        init_rate_half(self.rate2)  # 初始化卷积分支权重为0.5
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)# 设置为可学习参数
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)# 初始化偏置为0

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)# 应用转换层
        scaling = float(self.head_dim) ** -0.5# 缩放因子，用于自注意力计算
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride # 计算输出的高度和宽度

        pe = self.conv_p(position(h, w, x.is_cuda))# 生成位置编码
        # 为自注意力机制准备q, k, v
        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1: # 如果步长大于1，则对q和位置编码进行降采样
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe
       # 展开k和位置编码，准备自注意力计算
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out
         # 计算注意力权重
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)
          # 应用注意力权重
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)
        # 动态卷积核
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)
        # 将注意力分支和卷积分支的输出相加
        return self.rate1 * out_att + self.rate2 * out_conv


# 输入 N C H W,  输出 N C H W
# 导入必要的PyTorch模块
import torch
from torch import nn
from torch.nn import functional as F

class CoTAttention(nn.Module):
    # 初始化CoT注意力模块
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim  # 输入的通道数
        self.kernel_size = kernel_size  # 卷积核大小

        # 定义用于键(key)的卷积层，包括一个分组卷积，BatchNorm和ReLU激活
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 定义用于值(value)的卷积层，包括一个1x1卷积和BatchNorm
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 缩小因子，用于降低注意力嵌入的维度
        factor = 4
        # 定义注意力嵌入层，由两个卷积层、一个BatchNorm层和ReLU激活组成
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim//factor, 1, bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor, kernel_size*kernel_size*dim, 1)
        )

    def forward(self, x):
        # 前向传播函数
        bs, c, h, w = x.shape  # 输入特征的尺寸
        k1 = self.key_embed(x)  # 生成键的静态表示
        v = self.value_embed(x).view(bs, c, -1)  # 生成值的表示并调整形状

        y = torch.cat([k1, x], dim=1)  # 将键的静态表示和原始输入连接
        att = self.attention_embed(y)  # 生成动态注意力权重
        att = att.reshape(bs, c, self.kernel_size*self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # 计算注意力权重的均值并调整形状
        k2 = F.softmax(att, dim=-1) * v  # 应用注意力权重到值上
        k2 = k2.view(bs, c, h, w)  # 调整形状以匹配输出

        return k1 + k2  # 返回键的静态和动态表示的总和

import torch
import torch.nn as nn

# 定义一个基本的卷积模块，包括卷积、批归一化和ReLU激活
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # 定义卷积层
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # 条件性地添加批归一化层
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        # 条件性地添加ReLU激活函数
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)  # 应用卷积
        if self.bn is not None:
            x = self.bn(x)  # 应用批归一化
        if self.relu is not None:
            x = self.relu(x)  # 应用ReLU
        return x

# 定义ZPool模块，结合最大池化和平均池化结果
class ZPool(nn.Module):
    def forward(self, x):
        # 结合最大值和平均值
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

# 定义注意力门，用于根据输入特征生成注意力权重
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7  # 设定卷积核大小
        self.compress = ZPool()  # 使用ZPool模块
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)  # 通过卷积调整通道数

    def forward(self, x):
        x_compress = self.compress(x)  # 应用ZPool
        x_out = self.conv(x_compress)  # 通过卷积生成注意力权重
        scale = torch.sigmoid_(x_out)  # 应用Sigmoid激活
        return x * scale  # 将注意力权重乘以原始特征

# 定义TripletAttention模块，结合了三种不同方向的注意力门
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()  # 定义宽度方向的注意力门
        self.hc = AttentionGate()  # 定义高度方向的注意力门
        self.no_spatial = no_spatial  # 是否忽略空间注意力
        if not no_spatial:
            self.hw = AttentionGate()  # 定义空间方向的注意力门

    def forward(self, x):
        # 应用注意力门并结合结果
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  # 转置以应用宽度方向的注意力
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  # 还原转置
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # 转置以应用高度方向的注意力
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  # 还原转置
        if not self.no_spatial:
            x_out = self.hw(x)  # 应用空间注意力
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)  # 结合三个方向的结果
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)  # 结合两个方向的结果（如果no_spatial为True）
        return x_out

# 示例代码
import torch
from torch import nn



class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用5x5核的卷积层，应用深度卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 两组卷积层，分别使用1x7和7x1核，用于跨度不同的特征提取，均应用深度卷积
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        # 另外两组卷积层，使用更大的核进行特征提取，分别为1x11和11x1，也是深度卷积
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        # 使用最大尺寸的核进行特征提取，为1x21和21x1，深度卷积
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        # 最后一个1x1卷积层，用于整合上述所有特征提取的结果
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone() # 克隆输入x，以便之后与注意力加权的特征进行相乘
        attn = self.conv0(x) # 应用初始的5x5卷积

        # 应用1x7和7x1卷积，进一步提取特征
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        # 应用1x11和11x1卷积，进一步提取特征
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        # 应用1x21和21x1卷积，进一步提取特征
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2 # 将所有特征提取的结果相加

        attn = self.conv3(attn) # 应用最后的1x1卷积层整合特征

        return attn * u # 将原始输入和注意力加权的特征相乘，返回最终结果

import torch
# 定义EMA模块
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        # 设置分组数量，用于特征分组
        self.groups = factor
        # 确保分组后的通道数大于0
        assert channels // self.groups > 0
        # softmax激活函数，用于归一化
        self.softmax = nn.Softmax(-1)
        # 全局平均池化，生成通道描述符
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 水平方向的平均池化，用于编码水平方向的全局信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向的平均池化，用于编码垂直方向的全局信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # GroupNorm归一化，减少内部协变量偏移
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 1x1卷积，用于学习跨通道的特征
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3卷积，用于捕捉更丰富的空间信息
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入特征图进行分组处理
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # 应用水平和垂直方向的全局平均池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 通过1x1卷积和sigmoid激活函数，获得注意力权重
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 应用GroupNorm和注意力权重调整特征图
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        # 将特征图通过全局平均池化和softmax进行处理，得到权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # 通过矩阵乘法和sigmoid激活获得最终的注意力权重，调整特征图
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 将调整后的特征图重塑回原始尺寸
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义SAFM类，继承自nn.Module
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        # n_levels表示特征会被分割成多少个不同的尺度
        self.n_levels = n_levels
        # 每个尺度的特征通道数
        chunk_dim = dim // n_levels

        # Spatial Weighting：针对每个尺度的特征，使用深度卷积进行空间加权
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # Feature Aggregation：用于聚合不同尺度处理过的特征
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation：使用GELU激活函数
        self.act = nn.GELU()

    def forward(self, x):
        # x的形状为(B,C,H,W)，其中B是批次大小，C是通道数，H和W是高和宽
        h, w = x.size()[-2:]

        # 将输入特征在通道维度上分割成n_levels个尺度
        xc = x.chunk(self.n_levels, dim=1)

        out = []
        for i in range(self.n_levels):
            if i > 0:
                # 计算每个尺度下采样后的大小
                p_size = (h // 2**i, w // 2**i)
                # 对特征进行自适应最大池化，降低分辨率
                s = F.adaptive_max_pool2d(xc[i], p_size)
                # 对降低分辨率的特征应用深度卷积
                s = self.mfr[i](s)
                # 使用最近邻插值将特征上采样到原始大小
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                # 第一尺度直接应用深度卷积，不进行下采样
                s = self.mfr[i](xc[i])
            out.append(s)

        # 将处理过的所有尺度的特征在通道维度上进行拼接
        out = torch.cat(out, dim=1)
        # 通过1x1卷积聚合拼接后的特征
        out = self.aggr(out)
        # 应用GELU激活函数并与原始输入相乘，实现特征调制
        out = self.act(out) * x
        return out

import torch.nn as nn
import torch

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        # 通道注意力子模块
        self.channel_attention = nn.Sequential(
            # 降维，减少参数数量和计算复杂度
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),  # 非线性激活
            # 升维，恢复到原始通道数
            nn.Linear(int(in_channels / rate), in_channels)
        )

        # 空间注意力子模块
        self.spatial_attention = nn.Sequential(
            # 使用7x7卷积核进行空间特征的降维处理
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),  # 批归一化，加速收敛，提升稳定性
            nn.ReLU(inplace=True),  # 非线性激活
            # 使用7x7卷积核进行空间特征的升维处理
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)  # 批归一化
        )

    def forward(self, x):
        b, c, h, w = x.shape  # 输入张量的维度信息
        # 调整张量形状以适配通道注意力处理
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        # 应用通道注意力，并恢复原始张量形状
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        # 生成通道注意力图
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        # 应用通道注意力图进行特征加权
        x = x * x_channel_att

        # 生成空间注意力图并应用进行特征加权
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

# 示例代码：使用GAM_Attention对一个随机初始化的张量进行处理
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义PSA模块
class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4):
        super(PSA, self).__init__()
        self.S = S  # 尺度的数量，用于控制多尺度处理的维度

        # 定义不同尺度的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            for i in range(S)
        ])

        # 定义每个尺度对应的SE模块
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), # 自适应平均池化到1x1
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False), # 减少通道数
                nn.ReLU(inplace=True),# ReLU激活函数
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False), # 恢复通道数
                nn.Sigmoid()# Sigmoid激活函数，输出通道注意力权重
            ) for i in range(S)
        ])

        self.softmax = nn.Softmax(dim=1)  # 对每个位置的尺度权重进行归一化

    def forward(self, x):
        b, c, h, w = x.size()

         # 将输入在通道维度上分割为S份，对应不同的尺度
        SPC_out = x.view(b, self.S, c // self.S, h, w)

       # 对每个尺度的特征应用对应的卷积层
        conv_out = []
        for idx, conv in enumerate(self.convs):
            conv_out.append(conv(SPC_out[:, idx, :, :, :]))
        SPC_out = torch.stack(conv_out, dim=1)

        # 对每个尺度的特征应用对应的SE模块，获得通道注意力权重
        se_out = [se(SPC_out[:, idx, :, :, :]) for idx, se in enumerate(self.se_blocks)]
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand(-1, -1, -1, h, w)  # 扩展以匹配SPC_out的尺寸

        # 应用Softmax归一化注意力权重
        softmax_out = self.softmax(SE_out)

        # 应用注意力权重并合并多尺度特征
        PSA_out = SPC_out * softmax_out
        PSA_out = torch.sum(PSA_out, dim=1)  # 沿尺度维度合并特征

        return PSA_out
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


# 将设备和数据类型转换为字典格式

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

# 确保输入是元组形式
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

# 在指定维度上扩展张量
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 将相对位置编码转换为绝对位置编码
def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


# 生成一维的相对位置logits
def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits

# 相对位置嵌入类
class RelPosEmb(nn.Module):
    def __init__(
            self,
            block_size,
            rel_size,
            dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


# HaloAttention类

class HaloAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            block_size,
            halo_size,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size=block_size,
            rel_size=block_size + (halo_size * 2),
            dim_head=dim_head
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
         # 验证输入特征图维度是否符合要求
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device

        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)

        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c=c)

       #生成查询、键、值

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # 拆分头部

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))

        # 缩放查询向量

        q *= self.scale

        # 计算注意力

        sim = einsum('b i d, b j d -> b i j', q, k)

        # 添加相对位置偏置

        sim += self.rel_pos_emb(q)

        # 掩码填充

        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + (halo * 2), stride=block, padding=halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b=b, h=heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # 注意力机制

        attn = sim.softmax(dim=-1)

        # 聚合

        out = einsum('b i j, b j d -> b i d', attn, v)

        # 合并和组合头部

        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out = self.to_out(out)

        # 将块合并回原始特征图

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=(h // block), w=(w // block), p1=block,
                        p2=block)
        return out


# 输入 N C H W,  输出 N C H W
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        # 计算维度压缩后的向量长度
        self.d = max(L, channel // reduction)
        # 不同尺寸的卷积核组成的卷积层列表
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        # 通道数压缩的全连接层
        self.fc = nn.Linear(channel, self.d)
        # 为每个卷积核尺寸对应的特征图计算注意力权重的全连接层列表
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        # 注意力权重的Softmax层
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        # 通过不同尺寸的卷积核处理输入
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        # 将所有卷积核的输出求和得到融合特征图U
        U = sum(conv_outs)  # bs,c,h,w

        # 对融合特征图U进行全局平均池化，并通过全连接层降维得到Z
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        # 计算每个卷积核对应的注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weights = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1

        # 将注意力权重应用到对应的特征图上，并对所有特征图进行加权求和得到最终的输出V
        V = (attention_weights * feats).sum(0)
        return V


# class mposition(nn.Module):
#     def __init__(self,):
#         super().__init__()
#         self.multi_head_attention = RelationAwareMultiHeadAttention(hidden_size, num_heads=4, k=10, seq_len)
#
#
#
#     def forward(self,input):

class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        # self.softmax=nn.Softmax(dim=1)
        self.softmax=nn.Softmax(dim=2)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out

class ExternalAttention_four_dim(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ea = ExternalAttention(d_model=c, S=32)

    def forward(self,input):
        N = input.shape[2]* input.shape[3]
        C=input.shape[1]
        H=input.shape[2]
        W=input.shape[3]
        B=input.shape[0]
        input_tensor_reshaped = input.permute(0, 2, 3, 1).reshape(B, N, C)
        output=self.ea(input_tensor_reshaped)
        output_reshaped = output.view(B, H, W, C)  # (B, H, W, hidden_size)
        output_final = output_reshaped.permute(0, 3, 1, 2)  # (B, C, H, W)
        return output_final


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=1, kernel_size=3):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        # forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # input gate
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # output gate
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # initialize model
        self.init_hidden()

    def forward(self, x, h, c):

        """
        :param x: (1,21,h,w)
        :param h: (1,1,h,w)
        :param c: (1,1,h,w)
        :return: c, h
        """

        # # initialize if c,h is none
        # if h is None:
        #     h = torch.zeros(1, self.hidden_channels, x.shape[2], x.shape[3]).cuda()
        # if c is None:
        #     c = torch.zeros(1, self.hidden_channels, x.shape[2], x.shape[3]).cuda()


        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
try:
    from pytorch_wavelets import DWTForward  # 按照这个第三方库需要安装pip install pytorch_wavelets==1.3.0
    # 如果提示缺少pywt库则安装 pip install PyWavelets
except:
    pass


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x





    # Generating Sample image


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


from torch.nn.parameter import Parameter
class ShuffleAttention(nn.Module):
    # 初始化Shuffle Attention模块
    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G  # 分组数量
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，用于生成通道注意力
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  # 分组归一化，用于空间注意力
        # 以下为通道注意力和空间注意力的权重和偏置参数
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于生成注意力图

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 通道混洗方法，用于在分组处理后重组特征
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    # 前向传播方法
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)  # 将输入特征图按照分组维度进行重排

        x_0, x_1 = x.chunk(2, dim=1)  # 将特征图分为两部分，分别用于通道注意力和空间注意力

        # 通道注意力分支
        x_channel = self.avg_pool(x_0)  # 对第一部分应用全局平均池化
        x_channel = self.cweight * x_channel + self.cbias  # 应用学习到的权重和偏置
        x_channel = x_0 * self.sigmoid(x_channel)  # 通过sigmoid激活函数和原始特征图相乘，得到加权的特征图

        # 空间注意力分支
        x_spatial = self.gn(x_1)  # 对第二部分应用分组归一化
        x_spatial = self.sweight * x_spatial + self.sbias  # 应用学习到的权重和偏置
        x_spatial = x_1 * self.sigmoid(x_spatial)  # 通过sigmoid激活函数和原始特征图相乘，得到加权的特征图

        # 将通道注意力和空间注意力的结果沿通道维度拼接
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)  # 重新调整形状以匹配原始输入的维度

        # 应用通道混洗，以便不同分组间的特征可以交换信息
        out = self.channel_shuffle(out, 2)
        return out


# 输入 N C H W,  输出 N C H W
# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)
#     se = ShuffleAttention(channel=512, G=8)
#     output = se(input)
#     print(output.shape)
import torch
from torch import nn
import torch.nn.functional as F

# 用于调整数值，使其可以被某个除数整除，常用于网络层中通道数的设置。
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保减小的百分比不超过一定的比例（round_limit）
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

# Radix Softmax用于处理分组特征的归一化
class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality



    def forward(self, x):
        batch = x.size(0)
        # 根据radix是否大于1来决定使用softmax还是sigmoid进行归一化
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = x.sigmoid()
        return x

# SplitAttn模块实现分裂注意力机制
class SplitAttn(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        # 根据输入通道数、radix和rd_ratio计算注意力机制的中间层通道数
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        # 核心卷积层
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        # 后续层以及RadixSoftmax
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        # 卷积和激活
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        # 计算分裂注意力
        B, RC, H, W = x.shape
        if self.radix > 1:
            # 对特征进行重组和聚合
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        # 全局平均池化和两层全连接网络，应用RadixSoftmax
        x_gap = x_gap.mean(2, keepdims=True).mean(3, keepdims=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out




class sed(nn.Module):
    def __init__(self, channels=1):
        super(sed, self).__init__()
        inter_channels = 1
        self.sigmoid = nn.Sigmoid()
        self.local_attention1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_attention1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_attention2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_attention2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.f1=nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1),
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x,y):
        # x = input[0]
        # y = input[1]
        xuniony = y
        xa = x
        xl = self.local_attention1(xa)
        xg = self.global_attention1(xa)
        xl2 = self.local_attention2(xuniony)
        xg2 = self.global_attention2(xuniony)
        local=self.f1(torch.concat([xl, xl2], dim=1))
        glo=self.f2(torch.concat([xg, xg2], dim=1))


        # xlg = xl + xl2
        m1 = self.sigmoid(local)
        # xuniony = x * m1 + y * (1 - m1)
        # xlg2 = xg + xg2
        m2 = self.sigmoid(glo)

        z = (x+y) * m2 +(x+y) * (1 - m1)
        return z






import numpy as np
import torch
from torch import nn
from torch.nn import init


class ParNetAttention(nn.Module):
    # 初始化ParNet注意力模块
    def __init__(self, channel=512):
        super().__init__()
        # 使用自适应平均池化和1x1卷积实现空间压缩，然后通过Sigmoid激活函数产生权重图
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将空间维度压缩到1x1
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积，用于调整通道的权重
            nn.Sigmoid()  # Sigmoid函数，用于生成注意力图
        )

        # 通过1x1卷积实现特征重映射，不改变空间尺寸
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积，不改变特征图的空间尺寸
            nn.BatchNorm2d(channel)  # 批量归一化
        )
        # 通过3x3卷积捕获空间上下文信息
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),  # 3x3卷积，保持特征图尺寸不变
            nn.BatchNorm2d(channel)  # 批量归一化
        )
        self.silu = nn.SiLU()  # SiLU激活函数，也被称为Swish函数


    def forward(self, x):
        # x是输入的特征图，形状为(Batch, Channel, Height, Width)
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)  # 通过1x1卷积处理x
        x2 = self.conv3x3(x)  # 通过3x3卷积处理x
        x3 = self.sse(x) * x  # 应用空间压缩的注意力权重到x上
        y = self.silu(x1 + x2 + x3)  # 将上述三个结果相加并通过SiLU激活函数激活，获得最终输出
        return y

def Conv_Stage(input_dim, dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i + 1], kernel_size=3, bias=bias, padding=1),
            nn.BatchNorm2d(dim_list[i + 1]),
            nn.ReLU(inplace=True)
        )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 1, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)


def Conv_Stage2(input_dim, dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i + 1], kernel_size=3, bias=bias, padding=1),
            nn.BatchNorm2d(dim_list[i + 1]),
            nn.ReLU(inplace=True)
        )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 8, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)


class RB_riddnet(nn.Module):
    def __init__(self, cin=3):
        super(RB_riddnet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1_b = Conv_Stage(cin, [8, 4, 16])
        self.conv2_b = nn.Sequential(
            nn.Conv2d(cin, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.soft_boundary = Conv_Stage2(16, [8, 8, 8, 8], output_map=True)

    def forward(self, x):
        xf_1_b = self.conv1_b(x)  # (1, 16, 320, 320)
        xf_2_b = self.conv2_b(x)  # (1, 16, 320, 320)
        unet1 = torch.add(xf_1_b, xf_2_b)  # (1, 16, 320, 320)
        unet1 = self.soft_boundary(unet1)
        # unet1=self.relu(unet1)
        # boundary_soft = torch.softmax(unet1, 1)
        return unet1


def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert (h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h + h2, crop_w:crop_w + w2]
    return data


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class RBblock(nn.Module):
    def __init__(self, c_in):
        super(RBblock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 64, 1, stride=1)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 1, stride=1)
        self.conv4 = nn.Conv2d(c_in, 32, 1, stride=1)

    def forward(self, x):
        a1 = self.conv1(x)
        a2 = self.conv2(a1)
        a3 = self.conv3(a2)
        a4 = self.conv4(x)
        return a4 + a3



class lastblock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(lastblock,self).__init__()
        # self.rb_riddnet = RB_riddnet(c_in)
        self.rb = ECAAttention(kernel_size=3)

        # self.att=macnnet.Attention(c_in, c_in, LayerNorm_type='WithBias')
        # self.rate = rate
        #
        # self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # dilation = self.rate*1 if self.rate >= 1 else 1
        # self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu1 = nn.ReLU(inplace=True)
        # dilation = self.rate*2 if self.rate >= 1 else 1
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu2 = nn.ReLU(inplace=True)
        # dilation = self.rate*3 if self.rate >= 1 else 1
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu3 = nn.ReLU(inplace=True)
        # self._initialize_weights()

    def forward(self, x):
        # x=self.att(x)+x
        # x=self.rb_riddnet(x)
        x = self.rb(x)
        o = self.relu(self.conv(x))
        # o1 = self.relu1(self.conv1(o))
        # o2 = self.relu2(self.conv2(o))
        # o3 = self.relu3(self.conv3(o))
        # out = o + o1 + o2 + o3
        out = o
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class otherblock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(otherblock, self).__init__()
        self.att = CBAM(c_in)
        # self.conv = nn.Conv2d(c_in, 21, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(32, 21, 3, stride=1, padding=1)




    def forward(self, x):
        # x=self.att(x)+x
        # x=self.rb_riddnet(x)
        # x = self.att(x)
        x=self.att(x)
        # print(x.shape)
        o = self.relu(self.conv(x))


        out = self.conv1(o)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
class blockw(nn.Module):
    def __init__(self, c_in, rate=4):
        super(blockw, self).__init__()
        # self.rb_riddnet = RB_riddnet(c_in)
        # self.rb = macnnet.CBAM(c_in)

        self.rb=SimAM()
        # self.att=macnnet.Attention(c_in, c_in, LayerNorm_type='WithBias')
        # self.rate = rate
        #
        # self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = LeakyReLULayer(12,12)


        # dilation = self.rate*1 if self.rate >= 1 else 1
        # self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu1 = nn.ReLU(inplace=True)
        # dilation = self.rate*2 if self.rate >= 1 else 1
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu2 = nn.ReLU(inplace=True)
        # dilation = self.rate*3 if self.rate >= 1 else 1
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu3 = nn.ReLU(inplace=True)
        # self._initialize_weights()

    def forward(self, x):
        # x=self.att(x)+x
        # x=self.rb_riddnet(x)
        x = self.rb(x)
        o = self.relu(self.conv(x))
        # o1 = self.relu1(self.conv1(o))
        # o2 = self.relu2(self.conv2(o))
        # o3 = self.relu3(self.conv3(o))
        # out = o + o1 + o2 + o3
        out = o
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()



class one_up(nn.Module):
    def __init__(self,cin,cout):
        super(one_up,self).__init__()
        self.upsample_lsat = nn.ConvTranspose2d(cin, cin, 4, stride=2, bias=False)
        self.conv_last_1= nn.Conv2d(cin, cout, 3, stride=1, padding=1)
        self.relu_last_1 = nn.ReLU(inplace=True)
        self.conv_last_2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1)
        self.relu_last_2= nn.ReLU(inplace=True)



    def forward(self,x):
        o=self.upsample_lsat(x)
        o=self.relu_last_1(self.conv_last_1(o))
        out=self.relu_last_2(self.conv_last_2(o))

        return out

class last_up(nn.Module):
    def __init__(self):
        super(last_up,self).__init__()
        self.up1= one_up(512, 256)
        self.up2 = one_up(256, 128)
        self.up3 = one_up(128, 64)
        self.MSB = lastblock(64, 2)
        self.conv = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        # self.upsample_lsat_1 = nn.ConvTranspose2d(512, 512, 4, stride=2, bias=False)
        # self.conv_last_1_1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        # self.relu_last_1_1 = nn.ReLU(inplace=True)
        # self.conv_last_1_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.relu_last_1_2 = nn.ReLU(inplace=True)
        #
        # self.upsample_lsat_2 = nn.ConvTranspose2d(256, 256, 4, stride=2, bias=False)
        # self.conv_last_2_1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.relu_last_2_1 = nn.ReLU(inplace=True)
        # self.conv_last_1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.relu_last_2_2 = nn.ReLU(inplace=True)
        #
        # self.upsample_lsat_3 = nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        # self.conv_last_3_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.relu_last_3_1 = nn.ReLU(inplace=True)
        # self.conv_last_3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.relu_last_3_2 = nn.ReLU(inplace=True)
        #
        # self.upsample_lsat_4 = nn.ConvTranspose2d(64, 64, 4, stride=2, bias=False)
        # self.conv_last_4_1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        # self.relu_last_4_1 = nn.ReLU(inplace=True)
        # self.conv_last_4_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.relu_last_4_2 = nn.ReLU(inplace=True)



    def forward(self,x):

        x1=self.up1(x)

        x2=self.up2(x1)

        x3=self.up3(x2)




        out=self.MSB(x3)
        out = self.conv(out)

        return out










class daf_ES_Extractor(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=4):
        super(daf_ES_Extractor, self).__init__()
        # self.rb_riddnet = RB_riddnet()
        self.pretrain = pretrain
        t = 1
        # self.att=macnnet.Attention(3,3, LayerNorm_type='WithBias')
        self.features = VGG16_C(pretrain, logger)
        self.blockw1_1 = blockw(64, rate)
        self.otherblock1_1 = otherblock(64, rate)

        self.blockw1_2 = blockw(64, rate)
        self.otherblock1_2 = otherblock(64, rate)

        self.conv1_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)

        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.blockw2_1 = blockw(128, rate)
        self.blockw2_2 = blockw(128, rate)

        self.otherblock2_1 = otherblock(128, rate)
        self.otherblock2_2 = otherblock(128, rate)

        self.conv2_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.blockw3_1 = blockw(256, rate)
        self.blockw3_2 = blockw(256, rate)
        self.blockw3_3 = blockw(256, rate)

        self.otherblock3_1 = otherblock(256, rate)
        self.otherblock3_2 = otherblock(256, rate)
        self.otherblock3_3 = otherblock(256, rate)



        self.conv3_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.blockw4_1 = blockw(512, rate)
        self.blockw4_2 = blockw(512, rate)
        self.blockw4_3 = blockw(512, rate)

        self.otherblock4_1 = otherblock(512, rate)
        self.otherblock4_2 = otherblock(512, rate)
        self.otherblock4_3 = otherblock(512, rate)



        self.conv4_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.blockw5_1 = blockw(512, rate)
        self.blockw5_2 = blockw(512, rate)
        self.blockw5_3 = blockw(512, rate)
        self.otherblock5_1 = otherblock(512, rate)
        self.otherblock5_2 = otherblock(512, rate)
        self.otherblock5_3 = otherblock(512, rate)


        self.conv5_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)

        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)

        self.upsample_2_ = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4_ = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8_ = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5_ = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        #
        # self.lstm1=macnnet.ConvLSTMCell()
        # self.lstm2 = macnnet.ConvLSTMCell()
        # self.lstm3 = macnnet.ConvLSTMCell()
        # self.lstm4 = macnnet.ConvLSTMCell()
        # self.lstm5 = macnnet.ConvLSTMCell()

        # self.last_up=last_up()
        self.sed1=sed(channels=1)
        self.sed2=sed(channels=1)
        self.sed3=sed(channels=1)
        self.sed4=sed(channels=1)
        self.sed5=sed(channels=1)

        #
        # self.upsample_lsat_2 = nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False)
        #
        #
        # self.upsample_lsat_3 = nn.ConvTranspose2d(128, 64, 4, stride=2, bias=False)
        #
        #
        #
        # self.upsample_lsat_3 = nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False)

        # self.att11=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att21=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att31=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att41=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att51=macnnet.Attention(21, 21, LayerNorm_type='WithBias')

        # self.CGA1_1=CGAnet.CGAFusion(64)
        # self.CGA1_2 = CGAnet.CGAFusion(64)
        # self.CGA2_1 = CGAnet.CGAFusion(128)
        # self.CGA2_2 = CGAnet.CGAFusion(128)
        # self.CGA3_1 = CGAnet.CGAFusion(256)
        # self.CGA3_2 = CGAnet.CGAFusion(256)
        # self.CGA3_3 = CGAnet.CGAFusion(256)
        # self.CGA4_1 = CGAnet.CGAFusion(512)
        # self.CGA4_2 = CGAnet.CGAFusion(512)
        # self.CGA4_3 = CGAnet.CGAFusion(512)
        # self.CGA5_1 = CGAnet.CGAFusion(512)
        # self.CGA5_2 = CGAnet.CGAFusion(512)
        # self.CGA5_3 = CGAnet.CGAFusion(512)

        # self.fuse = nn.Conv2d(10, 1, 1, stride=1)l
        # self.fuse_1 = nn.Conv2d(10, 5, 1, stride=1)
        self.fuse = nn.Conv2d(5, 1, 1, stride=1)
        # self.fuse1 = nn.Conv2d(3, 1, 1, stride=1)
        # self.fuse2 = nn.Conv2d(3, 1, 1, stride=1)
        # self.fuse3 = nn.Conv2d(3, 1, 1, stride=1)
        # self.fuse4= nn.Conv2d(3, 1, 1, stride=1)

        # # wht
        self._initialize_weights(logger)

    def forward(self, x):
        # atten_score=self.rb_riddnet(x)
        features = self.features(x)
        #     1,c,h,w
        # for i in range(len(features)):
        #     print(f"features[{i}]size:  {features[i].shape}")
        sum1 = self.conv1_1_down(self.blockw1_1(features[0])) + \
               self.conv1_2_down(self.blockw1_2(features[1]))
        # msb(ciin,32)  down (32,21)
        sum11=self.otherblock1_1(features[0])+self.otherblock1_2(features[1])


        s1 = self.score_dsn1((sum1))
        s11 = self.score_dsn1_1(sum11)

        # print(s1.data.shape, s11.data.shape)
        sum2 = self.conv2_1_down(self.blockw2_1(features[2])) + \
               self.conv2_2_down(self.blockw2_2(features[3]))

        sum21=self.otherblock2_1(features[2])+self.otherblock2_2(features[3])
        s2 = self.score_dsn2((sum2))

        s21 = self.score_dsn2_1(sum21)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2_(s21)
        # print(s2.data.shape, s21.data.shape)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)

        sum3 = self.conv3_1_down(self.blockw3_1(features[4])) + \
               self.conv3_2_down(self.blockw3_2(features[5]))+self.conv3_3_down(self.blockw3_3(features[6]))

        sum31 =self.otherblock3_1(features[4])+self.otherblock3_2(features[5])+self.otherblock3_3(features[6])

        s3 = self.score_dsn3((sum3))
        s3 = self.upsample_4(s3)
        # print(s3.data.shape)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum31)
        # print(s31.shape)
        s31 = self.upsample_4_(s31)
        # print(s31.data.shape)
        s31 = crop(s31, x, 2, 2)

        sum4 = self.conv4_1_down(self.blockw4_1(features[7])) + \
               self.conv4_2_down(self.blockw4_2(features[8]))+ self.conv4_3_down(self.blockw4_3(features[9]))

        sum41 = self.otherblock4_1(features[7])+self.otherblock4_2(features[8])+self.otherblock4_3(features[9])
        s4 = self.score_dsn4((sum4))
        s4 = self.upsample_8(s4)
        # print(s4.data.shape)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum41)
        s41 = self.upsample_8_(s41)
        # print(s41.data.shape)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.blockw5_1(features[10])) + \
               self.conv5_2_down(self.blockw5_2(features[11]))+self.conv5_3_down(self.blockw5_3(features[12]))

        sum51 =  self.otherblock5_1(features[10])+self.otherblock5_2(features[11])+self.otherblock5_3(features[12])

        s5 = self.score_dsn5((sum5))
        s5 = self.upsample_8_5(s5)
        # print(s5.data.shape)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum51)
        s51 = self.upsample_8_5_(s51)
        # print(s51.data.shape)
        s51 = crop(s51, x, 0, 0)

        # print(s1.shape, s11.shape, s2.shape, s21.shape,s3.shape,s31.shape,s4.shape,s41.shape,s5.shape,s51.shape)
        # print(sum1.shape,sum2.shape,sum3.shape,sum4.shape,sum5.shape)

        s11 = torch.sigmoid(s11)
        s21 = torch.sigmoid(s21)
        s31 = torch.sigmoid(s31)
        s41 = torch.sigmoid(s41)
        s51 = torch.sigmoid(s51)

        # s1 = torch.sigmoid(s1)
        # s2 = torch.sigmoid(s2)
        # s3 = torch.sigmoid(s3)
        # s4 = torch.sigmoid(s4)
        # s5 = torch.sigmoid(s5)


        # p1 = (s1 * s11 )
        # p2 = (s2 * s21)
        # p3 = (s3 * s31)
        # p4 = (s4 * s41)
        # p5 = (s5 * s51)

        # p1 = (s1 * s11 + s1)
        # p2 = (s2 * s21 + s2)
        # p3 = (s3 * s31) + s3
        # p4 = (s4 * s41) + s4
        # p5 = (s5 * s51) + s5
        #
        # up_pic=self.last_up(features[-1])
        # up_pic = crop(up_pic, x, 0, 0)

        p1=self.sed1(s1,s11)
        p2=self.sed2(s2,s21)
        p3=self.sed3(s3,s31)
        p4=self.sed4(s4,s41)
        p5=self.sed5(s5,s51)

        fuse = self.fuse(torch.cat([p1, p2, p3, p4, p5], 1))

        p1=torch.sigmoid(p1)
        p2 = torch.sigmoid(p2)
        p3 = torch.sigmoid(p3)
        p4 = torch.sigmoid(p4)
        p5 = torch.sigmoid(p5)

        fuse=torch.sigmoid(fuse)

        return [p1, p2, p3, p4, p5,fuse],torch.cat([p1, p2, p3, p4, p5,fuse], 1)

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()

            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                # print(name)
                # print(len(name.split('.')[0]))
                if len(name.split('.'))<=2:
                    # print(name)
                    k = int(name.split('.')[0].split('_')[1])
                    param.copy_(get_upsampling_weight(1, 1, k * 2))



            elif 'fuse' in name:
                if logger:
                    logger.info('init params_autho %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant_(param, 0.080)
            else:
                if logger:
                    logger.info('init params_autho %s ' % name)
                if 'bias' in name:
                    param.zero_()
                if "weight" in name:
                    param.normal_(0, 0.01)
        # print self.conv1_1_down.weight

def bspline_wavelet(x, scale):
    return (1 / 6) * F.relu(scale*x)\
    - (8 / 6) * F.relu(scale*x - (1 / 2))\
    + (23 / 6) * F.relu(scale*x - (1))\
    - (16 / 3) * F.relu(scale*x - (3 / 2))\
    + (23 / 6) * F.relu(scale*x - (2))\
    - (8 / 6) * F.relu(scale*x - (5 / 2))\
    +(1 / 6) * F.relu(scale*x - (3))

class BSplineWavelet(nn.Module):
    def __init__(self, scale=torch.as_tensor(1)):
        super().__init__()
        self.scale = torch.as_tensor(scale)

    def forward(self, x):
        output = bspline_wavelet(x, self.scale)
        return output
class LeakyReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,negative_slope: float = 0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        return self.leaky_relu(x)

if __name__ == '__main__':
    a=torch.randn([8,3,32,32])
    model=daf_ES_Extractor()
    b=model(a)[0]
    for i in range(len(b)):
        print(b[i].shape)
