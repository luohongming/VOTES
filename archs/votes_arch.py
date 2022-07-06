import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

from utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer, default_init_weights
from .edvr_arch import PCDAlignment
from .bin_arch import BIN

from utils.img_util import tensor2img, rgb2gray_tensor

logger = logging.getLogger('basicsr')

@ARCH_REGISTRY.register()
class VOTES(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 num_modulation_blocks=5,
                 bin_model=None,
                 sr_factor=2):
        super(VOTES, self).__init__()

        self.sr_scale = sr_factor
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.bin = BIN(arch_scale=2, G0=64, D=6)
        if bin_model is not None:
            logger.info(f'Loading {self.bin.__class__.__name__} model from {bin_model}.')
            load_net = torch.load(bin_model, map_location=lambda storage, loc:storage)
            if 'params' in load_net:
                load_net = load_net['params']
            self.bin.load_state_dict(load_net, strict=True)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        self.fusion = MaskFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx,
                                 modulation_blocks=num_modulation_blocks)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        if sr_factor == 2:
            self.upconv1 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif sr_factor == 3:
            self.upconv1 = nn.Conv2d(num_feat, 64 * 9, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif sr_factor == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
            # self.pixel_shuffle
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')
        x_center = x[:, self.center_frame_idx, ...].contiguous()

        with torch.no_grad():
            _, x_center_blur = self.bin(x)
            x_center_blur = x_center_blur.clamp_(*(0, 1))
            x_center_gray = rgb2gray_tensor(x_center * 255.0).round()
            x_center_blur_gray = rgb2gray_tensor(x_center_blur * 255.0).round()
            diff = x_center_gray - x_center_blur_gray
            diff2 = torch.log2(torch.abs(diff) + 1)
            max_ = torch.max(diff2)
            error_map = ((diff2 / max_) * 255.0).round() / 255.0
        # error_map1 = mask[:, self.center_frame_idx, ...]
        # print(torch.sum(error_map - error_map1))
        # error_map = torch.zeros_like(error_map)


        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]

        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, ...].clone(), feat_l2[:, i, ...].clone(), feat_l3[:, i, ...].clone()
            ]

            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        feat = self.fusion(aligned_feat, error_map)

        out = self.reconstruction(feat)
        if self.sr_scale == 2 or self.sr_scale == 3:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        elif self.sr_scale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:
            raise ValueError(f'{self.sr_scale} is not available. ')

        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = F.interpolate(x_center, scale_factor=self.sr_scale, mode='bilinear', align_corners=False)
        out += base
        return out

class MaskFusion(nn.Module):

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2, modulation_blocks=5):
        super(MaskFusion, self).__init__()
        self.center_frame_idx = center_frame_idx

        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        self.mask_modulation_l = nn.ModuleList()
        self.feat_cas_l = nn.ModuleList()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.modualtion_blocks = modulation_blocks
        self.mask_modulation_first = nn.Conv2d(1, num_feat, 3, 1, 1)
        for i in range(modulation_blocks):
            self.feat_cas_l.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.mask_modulation_l.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))

        self.fusion = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, aligned_feat, mask):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
            mask (Tensor): Mask with shape (b, 1, h, w)

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()

        aligned_feat = aligned_feat.view(b, -1, h, w)
        # fusion
        feat1 = self.lrelu(self.feat_fusion(aligned_feat))

        # mask attention fusion

        mask_feat = self.mask_modulation_first(mask)
        for i in range(self.modualtion_blocks):
            feat1 = self.lrelu(self.feat_cas_l[i](feat1))
            mask_feat = self.lrelu(self.mask_modulation_l[i](mask_feat))
            feat1 = feat1 + mask_feat

        feat = self.fusion(feat1)

        return feat
