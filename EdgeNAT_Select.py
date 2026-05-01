import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from timm.models.layers import DropPath
from mmcv.runner import load_checkpoint
from mmcv.runner import BaseModule
from mmcv.utils import get_logger
from natten import NeighborhoodAttention2D as NeighborhoodAttention
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


def avg_max_reduce_hw_helper(x, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    max_pool = F.adaptive_max_pool2d(x, 1)

    if use_concat:
        res = torch.cat([avg_pool, max_pool], dim=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0])
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.cat(res, dim=1)


def mean_max_reduce_channel_helper(x, use_concat=True):
    # Reduce channel by mean and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True).values

    if use_concat:
        res = torch.cat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def mean_max_reduce_channel(x):
    # Reduce channel by mean and max
    # Return cat([mean_ch_0, max_ch_0, mean_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return mean_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return mean_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(mean_max_reduce_channel_helper(xi, False))
        return torch.cat(res, dim=1)


class FFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super(FFM, self).__init__()

        self.conv_l = nn.Sequential(
            nn.Conv2d(l_ch, h_ch, kernel_size=ksize, padding=ksize // 2, bias=False),
            ConditionalBatchNorm2d(h_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.resize_mode = resize_mode

    def check(self, l, h):
        assert l.dim() == 4 and h.dim() == 4
        l_h, l_w = l.shape[2:]
        h_h, h_w = h.shape[2:]
        assert l_h >= h_h and l_w >= h_w

    def prepare(self, l, h):
        l = self.prepare_l(l, h)
        h = self.prepare_h(l, h)
        return l, h

    def prepare_l(self, l, h):
        l = self.conv_l(l)
        return l

    def prepare_h(self, l, h):
        h_up = F.interpolate(h, size=l.shape[2:], mode=self.resize_mode)
        return h_up

    def fuse(self, l, h):
        out = l + h
        out = self.conv_out(out)
        return out

    def forward(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """
        self.check(l, h)
        l, h = self.prepare(l, h)
        out = self.fuse(l, h)
        return out


class FFM_ChAtten(FFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(l_ch, h_ch, out_ch, ksize, resize_mode)

        self.conv_lh_atten = nn.Sequential(
            nn.Conv2d(4 * h_ch, h_ch // 2, kernel_size=1, bias=False),
            ConditionalBatchNorm2d(h_ch // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(h_ch // 2, h_ch, kernel_size=1, bias=False),
            ConditionalBatchNorm2d(h_ch),
        )

    def fuse(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """
        atten = avg_max_reduce_hw([l, h])
        atten = torch.sigmoid(self.conv_lh_atten(atten))

        out = l * atten + h * (1 - atten)
        out = self.conv_out(out)
        return out


class FFM_SpAtten(FFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(l_ch, h_ch, out_ch, ksize, resize_mode)

        self.conv_lh_atten = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(1),
        )
        self._scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def fuse(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """
        atten = mean_max_reduce_channel([l, h])
        atten = torch.sigmoid(self.conv_lh_atten(atten))

        out = l * atten + h * (self._scale - atten)
        out = self.conv_out(out)
        return out


class FFM_SCAtten(FFM):
    """
    The UAFM with spatial and channel attention, which uses mean and max values.
    Args:
        l_ch (int): The channel of l tensor, which is the low level feature.
        h_ch (int): The channel of h tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for l tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling h tensor. Default: bilinear.
    """

    def __init__(self, l_ch, h_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(l_ch, h_ch, out_ch, ksize, resize_mode)

        self.conv_lh_s_atten = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(1),
        )
        self._scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.conv_lh_c_atten = nn.Sequential(
            nn.Conv2d(4 * h_ch, h_ch // 2, kernel_size=1, bias=False),
            ConditionalBatchNorm2d(h_ch // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(h_ch // 2, h_ch, kernel_size=1, bias=False),
            ConditionalBatchNorm2d(h_ch),
        )

        self.conv_sc_out = nn.Sequential(
            nn.Conv2d(h_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def fuse(self, l, h):
        """
        Args:
            l (Tensor): The low level feature.
            h (Tensor): The high level feature.
        """

        atten_s = mean_max_reduce_channel([l, h])
        atten_s = torch.sigmoid(self.conv_lh_s_atten(atten_s))

        out_s = l * atten_s + h * (self._scale - atten_s)

        atten_c = avg_max_reduce_hw([l, h])
        atten_c = torch.sigmoid(self.conv_lh_c_atten(atten_c))

        out_c = l * atten_c + h * (1 - atten_c)

        out = torch.cat((out_s, out_c), dim=1)
        out = self.conv_sc_out(out)

        return out


class ConvTokenizer(BaseModule):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(BaseModule):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(BaseModule):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(BaseModule):
    def __init__(
            self,
            dim,
            num_heads,
            kernel_size=7,
            dilation=None,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(BaseModule):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            kernel_size,
            dilations=None,
            downsample=True,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class EdgeNAT_MLA(BaseModule):
    def __init__(self, backbone_out_chs, backbone_indices, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode, fpn_Bu=False, align_corners=False):
        super(EdgeNAT_MLA, self).__init__()

        self.align_corners = align_corners
        self.fpn_Bu = fpn_Bu
        self.arm_type = arm_type
        self.cm_bin_sizes = cm_bin_sizes

        if self.cm_bin_sizes is not None:
            self.cm = ContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch, cm_bin_sizes)
        if self.arm_type is not None:

            arm_class = eval(arm_type)

            self.arm_list = nn.ModuleList()  # [..., arm8, arm16, arm32]
            for i in range(len(backbone_indices)):
                low_chs = backbone_out_chs[backbone_indices[i]]
                high_ch = cm_out_ch if i == (len(backbone_indices) - 1) else backbone_out_chs[backbone_indices[i + 1]]
                out_ch = backbone_out_chs[backbone_indices[i]]
                arm = arm_class(
                    low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
                self.arm_list.append(arm)

        self.deconvs = nn.ModuleList()
        for i in range(len(backbone_indices) - 1):
            self.deconvs.append(
                nn.Sequential(
                    nn.Conv2d(backbone_out_chs[i + 1], backbone_out_chs[i], kernel_size=3, stride=1, padding=1,
                              bias=False),
                    ConditionalBatchNorm2d(backbone_out_chs[i]),
                    nn.ReLU(inplace=True),
                    Upsample(scale_factor=2, mode=resize_mode, align_corners=align_corners)
                )
            )
        if self.arm_type is not None:
            self.out_convs = nn.ModuleList()
            for i in range(len(backbone_indices)):
                self.out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i] * 3, backbone_out_chs[i], kernel_size=3, stride=1, padding=1,
                                  bias=False),
                        ConditionalBatchNorm2d(backbone_out_chs[i]),
                        nn.ReLU(inplace=True),
                    )
                )
        else:
            self.out_convs = nn.ModuleList()
            for i in range(len(backbone_indices)):
                self.out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i] * 2, backbone_out_chs[i], kernel_size=3, stride=1, padding=1,
                                  bias=False),
                        ConditionalBatchNorm2d(backbone_out_chs[i]),
                        nn.ReLU(inplace=True),
                    )
                )

        if self.fpn_Bu:
            self.convs_Bu = nn.ModuleList()
            for i in range(len(backbone_indices) - 1):
                self.convs_Bu.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i], backbone_out_chs[i + 1], kernel_size=3, stride=2, padding=1,
                                  bias=False),
                        ConditionalBatchNorm2d(backbone_out_chs[i + 1]),
                        nn.ReLU(inplace=True),
                    )
                )

            self.out_convs_Bu = nn.ModuleList()
            for i in range(len(backbone_indices)):
                self.out_convs_Bu.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_out_chs[i] * 2, backbone_out_chs[i], kernel_size=3, stride=1, padding=1,
                                  bias=False),
                        ConditionalBatchNorm2d(backbone_out_chs[i]),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, in_feat_list):
        if self.cm_bin_sizes is not None:
            cm_out = self.cm(in_feat_list[-1])
        else:
            cm_out = in_feat_list[-1]
        out_feat_list = []
        if self.arm_type is not None:
            high_feat = cm_out
            for i in reversed(range(len(in_feat_list))):
                low_feat = in_feat_list[i]
                arm = self.arm_list[i]
                arm_feat = arm(low_feat, high_feat)
                if low_feat.shape == high_feat.shape:
                    high_feat = torch.cat((arm_feat, high_feat, low_feat), dim=1)
                else:
                    down_feat = self.deconvs[i](high_feat)
                    high_feat = torch.cat((arm_feat, down_feat, low_feat), dim=1)
                high_feat = self.out_convs[i](high_feat)
                out_feat_list.insert(0, high_feat)
        else:
            for i in reversed(range(len(in_feat_list))):
                if i == len(in_feat_list) - 1:
                    if self.cm_bin_sizes is not None:
                        feat = torch.cat((in_feat_list[i], cm_out), dim=1)
                        feat = self.out_convs[i](feat)
                    else:
                        feat = in_feat_list[i]
                else:
                    Td_feat = self.deconvs[i](feat)
                    feat = torch.cat((in_feat_list[i], Td_feat), dim=1)
                    feat = self.out_convs[i](feat)
                out_feat_list.insert(0, feat)

        if self.fpn_Bu:
            for i in range(len(in_feat_list)):
                if i > 0:
                    Bu_feat = self.convs_Bu[i - 1](feat)
                    feat = torch.cat((in_feat_list[i], Bu_feat), dim=1)
                    feat = self.out_convs_Bu[i](feat)
                else:
                    feat = in_feat_list[i]
                out_feat_list.append(feat)
        return out_feat_list

class ContextModule(BaseModule):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super(ContextModule, self).__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = nn.Sequential(
            nn.Conv2d(
                in_channels=inter_channels * len(bin_sizes) + in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1),
            ConditionalBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            ConditionalBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]
        r = input

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out = torch.cat((out, x), dim=1)

        out = torch.cat((out, r), dim=1)
        out = self.conv_out(out)
        return out


class EDHead(BaseModule):
    def __init__(self, idxs, in_chan, backbone_out_chs, resize_mode, align_corners=False):
        super(EDHead, self).__init__()

        self.align_corners = align_corners

        self.convs = nn.ModuleList()
        for idx in reversed(range(idxs)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_chan, backbone_out_chs[idx], kernel_size=3, stride=1, padding=1, bias=False),
                    ConditionalBatchNorm2d(backbone_out_chs[idx]),
                    nn.ReLU(inplace=True),
                )
            )
            in_chan = backbone_out_chs[idx]

        self.up = Upsample(scale_factor=2 ** idxs, mode=resize_mode, align_corners=align_corners)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x


# @BACKBONES.register_module()
class EdgeNAT_ES_Extraction(BaseModule):
    def __init__(
            self,
            embed_dim=192,
            mlp_ratio=2.0,
            depths=[3, 4, 18, 5],
            num_heads=[6, 12, 24, 48],
            drop_path_rate=0.3,
            in_chans=3,
            kernel_size=7,
            dilations=[[1, 20, 1], [1, 5, 1, 10], [1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5],
                       [1, 2, 1, 2, 1]],
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            frozen_stages=-1,
            pretrained=None,
            layer_scale=None,
            num_classes=1,
            backbone_indices=[0, 1, 2, 3],
            arm_type='FFM_SCAtten',
            cm_bin_sizes=[1, 2, 3, 6],
            cm_out_ch=1536,
            in_patch_size=4,
            fpn_Bu=False,
            resize_mode='bilinear'
    ):
        super(EdgeNAT_ES_Extraction, self).__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.features_chan = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.backbone_indices = backbone_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.features_chan[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.initWeights(pretrained)

        self.fedter_head = EdgeNAT_MLA(self.features_chan,
                                       self.backbone_indices,
                                       cm_bin_sizes,
                                       cm_out_ch,
                                       arm_type,
                                       resize_mode,
                                       fpn_Bu=fpn_Bu,
                                       )

        arm_out_chs = [self.features_chan[i] for i in self.backbone_indices]
        if fpn_Bu:
            arm_out_chs = arm_out_chs + arm_out_chs
        self.ed_heads = nn.ModuleList()
        for idx, in_ch in enumerate(arm_out_chs):
            idx = idx % len(self.backbone_indices)
            self.ed_heads.append(EDHead(self.backbone_indices[idx], in_ch, self.features_chan, resize_mode))
        self.fuse=EdgeNAT_SCAMLAHead(embed_dim)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(EdgeNAT_ES_Extraction, self).train(mode)
        self._freeze_stages()

    def initWeights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        outs = self.forward_tokens(x)
        feats_selected = [outs[i] for i in self.backbone_indices]

        feats = self.fedter_head(feats_selected)

        logit_list = []
        for x, ed_head in zip(feats, self.ed_heads):
            x = ed_head(x)
            logit_list.append(x)

        output=self.fuse(logit_list)

        return  output,output

class EdgeNAT_SCAMLAHead(BaseModule):
    def __init__(self, in_channels):
        super(EdgeNAT_SCAMLAHead, self).__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            ConditionalBatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2,  1, kernel_size=1, bias=False),
        )

        self.up = Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.conv_out(x)
        x = self.up(x)
        edge = torch.sigmoid(x)
        return edge

class EdgeNAT_Edge_EES_Extractor(BaseModule):
    def __init__(
            self,
            embed_dim=192,
            mlp_ratio=2.0,
            depths=[3, 4, 18, 5],
            num_heads=[6, 12, 24, 48],
            drop_path_rate=0.3,
            in_chans=3,
            kernel_size=7,
            dilations=[[1, 20, 1], [1, 5, 1, 10], [1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5],
                       [1, 2, 1, 2, 1]],
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            frozen_stages=-1,
            pretrained=None,
            layer_scale=None,
            num_classes=1,
            backbone_indices=[0, 1, 2, 3],
            arm_type='FFM_SCAtten',
            cm_bin_sizes=[1, 2, 3, 6],
            cm_out_ch=1536,
            in_patch_size=4,
            fpn_Bu=False,
            resize_mode='bilinear'
    ):
        super(EdgeNAT_Edge_EES_Extractor, self).__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.features_chan = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.backbone_indices = backbone_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.features_chan[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.initWeights(pretrained)

        self.fedter_head = EdgeNAT_MLA(self.features_chan,
                                       self.backbone_indices,
                                       cm_bin_sizes,
                                       cm_out_ch,
                                       arm_type,
                                       resize_mode,
                                       fpn_Bu=fpn_Bu,
                                       )

        arm_out_chs = [self.features_chan[i] for i in self.backbone_indices]
        if fpn_Bu:
            arm_out_chs = arm_out_chs + arm_out_chs
        self.ed_heads = nn.ModuleList()
        for idx, in_ch in enumerate(arm_out_chs):
            idx = idx % len(self.backbone_indices)
            self.ed_heads.append(EDHead(self.backbone_indices[idx], in_ch, self.features_chan, resize_mode))
        self.fuse_pre=EdgeNAT_SCAMLAHead_EES(embed_dim,embed_dim//2)
        self.fuse_final = nn.Conv2d(embed_dim // 2,1,1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.network[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(EdgeNAT_Edge_EES_Extractor, self).train(mode)
        self._freeze_stages()

    def initWeights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x):
        B,_,H,W=x.shape
        x = self.forward_embeddings(x)
        outs = self.forward_tokens(x)
        feats_selected = [outs[i] for i in self.backbone_indices]

        feats = self.fedter_head(feats_selected)

        logit_list = []
        for x, ed_head in zip(feats, self.ed_heads):
            x = ed_head(x)
            logit_list.append(x)

        output=self.fuse_pre(logit_list)

        Feature=torch.cat([torch.zeros(B, 1, H, W).to(output.device),torch.sigmoid(output),torch.ones(B, 1, H, W).to(output.device)],dim=1)
        output=torch.sigmoid(self.fuse_final(output))

        return  output,Feature

class EdgeNAT_SCAMLAHead_EES(BaseModule):
    def __init__(self, in_channels,out):
        super(EdgeNAT_SCAMLAHead_EES, self).__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            ConditionalBatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            ConditionalBatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2,  out, kernel_size=1, bias=False),
        )

        self.up = Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.conv_out(x)
        x = self.up(x)
        return x

class ConditionalBatchNorm2d(nn.BatchNorm2d):

    def forward(self, input):

        B, C, H, W = input.size()
        is_problematic_case_for_training = (B == 1) and (H == 1) and (W == 1)

        if self.training and is_problematic_case_for_training:
            return input
        else:
            output = super().forward(input)
            return output

if __name__ == '__main__':
    model = EdgeNAT_Edge_EES_Extractor()
    dummy_input = torch.rand(8, 3, 32, 32)
    output = model(dummy_input)
    print(output[0].shape,output[1].shape)
