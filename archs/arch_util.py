
import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from utils import get_root_logger

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, groups=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True, groups=groups)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """


    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class ResidualMaskBlock(nn.Module):
    """Residual mask block without BN.

        It has a style of:
            ---Conv-ReLU-Conv-*mask+-
             |_____________________|

        Args:
            num_feat (int): Channel number of intermediate features.
                Default: 64.
            res_scale (float): Residual scale. Default: 1.
            pytorch_init (bool): If set to True, use pytorch default init,
                otherwise, use default_init_weights. Default: False.
        """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualMaskBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.mask_conv1_1 = nn.Conv2d(1, num_feat, 3, 1, 1, bias=True)
        self.mask_conv1_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.mask_conv2_1 = nn.Conv2d(1, num_feat, 3, 1, 1, bias=True)
        self.mask_conv2_2 = nn.Conv2d(num_feat, 1, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, mask):
        identity = x
        mask_beta = self.mask_conv1_2(self.relu(self.mask_conv1_1(mask)))
        # mask_beta = self.sigmoid(mask_beta)
        mask_alpha = self.mask_conv2_2(self.relu(self.mask_conv2_1(mask)))
        mask_alpha = self.sigmoid(mask_alpha)

        out = self.conv2(self.relu(self.conv1(x))) * (1 + mask_alpha) + mask_beta
        return identity + out * self.res_scale

class ConvMaskBlock(nn.Module):
    def __init__(self, num_feat=64, pytorch_init=False):
        super(ConvMaskBlock, self).__init__()
        # self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.mask_conv1_1 = nn.Conv2d(1, num_feat, 3, 1, 1, bias=True)
        self.mask_conv1_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # self.mask_conv2_1 = nn.Conv2d(1, num_feat, 3, 1, 1, bias=True)
        # self.mask_conv2_2 = nn.Conv2d(num_feat, 1, 3, 1, 1, bias=True)
        # self.sigmoid = nn.Sigmoid()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x, mask):
        # identity = x
        mask_beta = self.mask_conv1_2(self.relu(self.mask_conv1_1(mask)))
        # mask_beta = self.sigmoid(mask_beta)
        # mask_alpha = self.mask_conv2_2(self.relu(self.mask_conv2_1(mask)))
        # mask_alpha = self.sigmoid(mask_alpha)

        out = self.conv2(self.relu(self.conv1(x))) + mask_beta
        return out * self.res_scale

class MaskFeatExtract(nn.Module):
    def __init__(self, num_feat=64, num_extract_block=5):
        super(MaskFeatExtract, self).__init__()
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.num_extract_block = num_extract_block
        self.mask_modulation_l = nn.ModuleList()
        self.feat_cas_l = nn.ModuleList()
        for i in range(num_extract_block):
            self.feat_cas_l.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.mask_modulation_l.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))

        self.fusion = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


    def forward(self, x, mask):

        feat1 = x.clone()
        mask1 = mask
        for i in range(self.num_extract_block):
            feat1 = self.lrelu(self.feat_cas_l[i](feat1))
            mask1 = self.lrelu(self.mask_modulation_l[i](mask1))
            feat1 = feat1 + mask1

        feat = self.fusion(feat1) + x

        return feat


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

