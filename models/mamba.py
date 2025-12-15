import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from PIL import Image
from torchvision import transforms
from functools import partial
from models.pre import ConvBNReLUBlock,PatchExpand,ChannelReductionBlock

# mamba out结构  除去了SSM的mamba结构
class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partical channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(self, dim, expension_ratio=8 / 3, kernel_size=7, conv_ratio=1.0,norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer=nn.GELU,drop_path=0.,**kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(p = 0.5)


    def forward(self, x):
        x = x.to(self.norm.weight.device)
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        #c = self.BN(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        return x + shortcut

# 用mamba out进行合并
# class Merge(nn.Module):
#     # 第一个参数dim1 为了patch expanding服务，提供通道数
#     # 第二个参数dim  为了mamba out块服务，提供的是输入的尺寸
#     def __init__(self,dim1,dim):
#
#         super(Merge, self).__init__()
#         self.dim = dim
#         self.dim1= dim1
#         self.block1 = GatedCNNBlock(dim[0])
#         self.block2 = GatedCNNBlock(dim[1])
#         self.block3 = GatedCNNBlock(dim[2])
#         self.block4 = GatedCNNBlock(dim[3])
#         self.patch_expand1 = PatchExpand((32, 32), self.dim1[0], dim_scale=2, norm_layer=nn.LayerNorm)
#         self.patch_expand2 = PatchExpand((64, 64), self.dim1[1], dim_scale=2, norm_layer=nn.LayerNorm)
#         self.patch_expand3 = PatchExpand((128, 128), self.dim1[2], dim_scale=2, norm_layer=nn.LayerNorm)
#         self.changechannellayer = ChannelReductionBlock(dim1[3], 3)
#         self.channel_change = ChannelReductionBlock(64,16)
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self,list):
#
#         x1 = list[3]
#         x1 = self.block1(x1)
#         #x1 = self.dropout(x1)
#         x1 = x1.permute(0, 2, 3, 1)
#         x1 = self.patch_expand1(x1)
#         x1 = x1.permute(0, 3, 2, 1)
#
#         x2 = torch.cat((x1,list[2]),dim=1)
#         x2 = self.block2(x2)
#         #x2 = self.dropout(x2)
#         x2 = x2.permute(0,2,3,1)
#         x2 = self.patch_expand2(x2)
#         x2 = x2.permute(0, 3, 2, 1)
#
#         x3 = torch.cat((x2,list[1]),dim=1)
#         x3 = self.block3(x3)
#         #x3 = self.dropout(x3)
#         x3 = x3.permute(0,2,3,1)
#         x3 = self.patch_expand3(x3)
#         x3 = x3.permute(0, 3, 2, 1)
#
#
#         x4 = torch.cat((x3,list[0]),dim=1)
#         x4 = self.block4(x4)
#         #x4 = self.dropout(x4)
#         x4 = self.channel_change(x4)
#         return x4


class Merge(nn.Module):
    # 第一个参数dim1 为了patch expanding服务，提供通道数
    # 第二个参数dim  为了mamba out块服务，提供的是输入的尺寸
    def __init__(self,dim):

        super(Merge, self).__init__()
        self.dim = dim
        self.block1 = ConvBNReLUBlock(dim[0],dim[0])
        self.block2 = ConvBNReLUBlock(dim[1],dim[1])
        self.block3 = ConvBNReLUBlock(dim[2],dim[2])
        self.block4 = ConvBNReLUBlock(dim[3],dim[3])
        self.patch_expand1 = PatchExpand((32, 32), self.dim[0], dim_scale=2, norm_layer=nn.LayerNorm)
        self.patch_expand2 = PatchExpand((64, 64), self.dim[1], dim_scale=2, norm_layer=nn.LayerNorm)
        self.patch_expand3 = PatchExpand((128, 128), self.dim[2], dim_scale=2, norm_layer=nn.LayerNorm)
        self.changechannellayer = ChannelReductionBlock(self.dim[3], 3)
        self.channel_change = ChannelReductionBlock(64,16)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,list):

        x1 = list[3]
        x1 = self.block1(x1)
        #x1 = self.dropout(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.patch_expand1(x1)
        x1 = x1.permute(0, 3, 2, 1)

        x2 = torch.cat((x1,list[2]),dim=1)
        x2 = self.block2(x2)
        #x2 = self.dropout(x2)
        x2 = x2.permute(0,2,3,1)
        x2 = self.patch_expand2(x2)
        x2 = x2.permute(0, 3, 2, 1)

        x3 = torch.cat((x2,list[1]),dim=1)
        x3 = self.block3(x3)
        #x3 = self.dropout(x3)
        x3 = x3.permute(0,2,3,1)
        x3 = self.patch_expand3(x3)
        x3 = x3.permute(0, 3, 2, 1)


        x4 = torch.cat((x3,list[0]),dim=1)
        x4 = self.block4(x4)
        #x4 = self.dropout(x4)
        #x4 = self.channel_change(x4)
        return x4