import torch
import torch.nn as nn
import time
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_
from models.mamba import GatedCNNBlock
from models.pre import StemLayer, PatchMerging2D,ConvBNReLUBlock, ChannelReductionBlock, resblock
import gc
from PIL import Image
import os
import matplotlib.pyplot as plt
from models.resnet import  *
from thop import profile
# 编码器结构  其中的dims中存储的就是四个不同尺度
# 编码器输出尺寸： torch.Size([1, 256, 256, 32]) torch.Size([1, 32, 256, 256]) torch.Size([1, 64, 128, 128]) torch.Size([1, 128, 64, 64]) torch.Size([1, 256, 32, 32])
# 首先把stem输出的结果改变一下，改为24   然后按照通道进行划分为  6   18
# 将B 6 256 256以及B 18 256 256分别放入编码器 最后得到四个尺度
class Encoder(nn.Module):
    def __init__(self,
                 dims=[32, 64, 128, 256],
                 norm_layer=nn.LayerNorm, ):
        super(Encoder, self).__init__()
        self.block1 = GatedCNNBlock(dims[3])
        self.block2 = GatedCNNBlock(dims[2])
        self.block3 = GatedCNNBlock(dims[1])
        self.block4 = GatedCNNBlock(dims[0])
        self.PatchMerging1 = PatchMerging2D(16, norm_layer=norm_layer)  # 24 就是Stem之后256的通道数，之后的依次加倍
        self.PatchMerging2 = PatchMerging2D(32, norm_layer=norm_layer)
        self.PatchMerging3 = PatchMerging2D(64, norm_layer=norm_layer)

    def forward(self, x):
        x_ret = []
        x1 = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        x1 = self.block1(x1)
        x_ret.append(x1)
        x1 = x1.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x1 = self.PatchMerging1(x1)
        x1 = x1.permute(0, 3, 1, 2)  # B H W C -> B C H W

        x2 = self.block2(x1)
        x_ret.append(x2)
        x2 = x2.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x2 = self.PatchMerging2(x2)
        x2 = x2.permute(0, 3, 1, 2)  # B H W C -> B C H W

        x3 = self.block3(x2)
        x_ret.append(x3)
        x3 = x3.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x3 = self.PatchMerging3(x3)
        x3 = x3.permute(0, 3, 1, 2)

        x4 = self.block4(x3)
        x_ret.append(x4)

        return x_ret


# 解码器结构  比如传入一个dim参数，存一个列表
# dim列表的值依次为第一次通道融合+上采样之后的通道数   第二次通道融合+上采样之后的通道数   第三次通道融合+上采样之后的通道数   第四次通道融合之后的通道数

class ToTal_decoder(nn.Module):
    def __init__(self,start_channel):
        super(ToTal_decoder,self).__init__()
        self.resblock1 = resblock(start_channel,start_channel)
        self.conv1 = nn.Conv2d(in_channels=start_channel, out_channels=start_channel // 2, kernel_size=(3, 3),stride=(1, 1),padding=1)
        self.BN1 = nn.BatchNorm2d(start_channel // 2)
        self.resblock2 = resblock(start_channel // 2, start_channel // 2)
        self.conv2 = nn.Conv2d(in_channels=start_channel // 2, out_channels=start_channel // 2, kernel_size=(3, 3),stride=(1, 1),padding=1)
        self.BN2 = nn.BatchNorm2d(start_channel // 2)
        self.resblock3 = resblock(start_channel // 2, start_channel // 2)
        self.conv3 = nn.Conv2d(in_channels=start_channel // 2, out_channels=start_channel // 2, kernel_size=(1, 1),stride=(1, 1))
        self.ReLU = nn.ReLU()
    def forward(self,x):
        x = self.resblock1(x)
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.ReLU(x)
        x = self.resblock2(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.ReLU(x)
        x = self.resblock3(x)
        x = self.conv3(x)
        return x
# 输入尺寸是torch.Size([1, 8, 256, 256])
# 输出尺寸是torch.Size([1, 32, 128, 128])


class Rebuilt(nn.Module):
    def __init__(self,channel1,channel2):
        super(Rebuilt,self).__init__()
        self.resblock1 = resblock(channel1,channel1)
        self.conv1 = nn.Conv2d(in_channels=channel2,out_channels=channel2,kernel_size=(3,3),stride=(1,1),padding=1)
        self.BN1 = nn.BatchNorm2d(channel2)
        self.change_channel = ChannelReductionBlock(channel2 + channel1,(channel2 + channel1)//2)
        self.resblock2 = resblock((channel2 + channel1)//2,(channel2 + channel1)//2)
        self.conv2 = nn.Conv2d(in_channels=(channel2 + channel1)//2,out_channels=(channel2 + channel1)//4,kernel_size=(3,3),stride=(1,1),padding=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
    def forward(self,x,y):
        x = self.resblock1(x)
        y = self.conv1(y)
        y = self.BN1(y)
        y = self.ReLU(y)
        z = torch.cat((x,y),dim=1)
        z = self.change_channel(z)
        z = self.resblock2(z)
        z = self.conv2(z)
        return z

class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.SA(x)
        return A

class GetMap(nn.Module):
    def __init__(self,x_channel):
        super(GetMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=x_channel,out_channels=x_channel//2,kernel_size=(3,3),stride=(1,1),padding=1)
        self.BN1 = nn.BatchNorm2d(x_channel//2)
        self.conv2 = nn.Conv2d(in_channels=x_channel//2,out_channels=x_channel//4,kernel_size=(3,3),stride=(1,1),padding=1)
        self.BN2 = nn.BatchNorm2d(x_channel//4)
        # self.conv3 = nn.Conv2d(in_channels=x_channel // 4, out_channels=x_channel // 2, kernel_size=(3, 3),stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=x_channel // 4, out_channels=x_channel // 4, kernel_size=(1, 1),stride=(1, 1))
        self.ReLU = nn.ReLU()
        self.attenion = Space_Attention(x_channel, x_channel//4)
        self.resblock = resblock(x_channel//2,x_channel//2)
    def forward(self,x1,x2,x3,x4):
        # x1是第一个图某个尺度公有的，x2是第二个图某个尺度公有的，x3是第一个图某个尺度私有的，x4是第二个图某个尺度私有的
        # 拼接之前 每个输入是否要经过一个卷机层  看最后的结果如何吧 再决定要不要加
        x = torch.cat((x1,x2,x3,x4),dim=1)
        x1 = x
        x2 = x
        x1 = self.conv1(x1)
        x1 = self.BN1(x1)
        x1 = self.ReLU(x1)
        x1 = self.resblock(x1)
        x1 = self.conv2(x1)
        x1 = self.BN2(x1)
        x1 = self.ReLU(x1)
        x1 = self.conv4(x1)

        x2 = self.attenion(x2)

        out = x1 * x2

        return out

class GetChangeMap(nn.Module):
    def __init__(self,x1_channel,x2_channel,x3_channel,x4_channel):
        super(GetChangeMap,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=x4_channel,out_channels=x3_channel,kernel_size=(3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.BN1 = nn.BatchNorm2d(x3_channel)
        self.conv2 = nn.ConvTranspose2d(in_channels=x3_channel*2,out_channels=x2_channel,kernel_size=(3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.BN2 = nn.BatchNorm2d(x2_channel)
        self.conv3 = nn.ConvTranspose2d(in_channels=x2_channel*2,out_channels=x1_channel,kernel_size=(3,3),stride=(2,2),padding=(1,1),output_padding=(1,1))
        self.BN3 = nn.BatchNorm2d(x1_channel)
        self.conv4 = nn.Conv2d(in_channels=x1_channel*2,out_channels=x1_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.BN4 = nn.BatchNorm2d(x1_channel)
        self.conv5 = nn.Conv2d(in_channels=x1_channel,out_channels=x1_channel,kernel_size=(1,1),stride=(1,1))
        self.BN5 = nn.BatchNorm2d(x1_channel)
        self.conv6 = nn.Conv2d(in_channels=x1_channel, out_channels=2, kernel_size=(3, 3), stride=(1, 1),padding=1)
        self.ReLU = nn.ReLU()

    def forward(self,x1,x2,x3,x4):
        x4 = self.conv1(x4)
        x4 = self.BN1(x4)
        x4 = self.ReLU(x4)


        x3 = torch.cat((x3,x4),dim=1)
        x3 = self.conv2(x3)
        x3 = self.BN2(x3)
        x3 = self.ReLU(x3)


        x2 = torch.cat((x2,x3),dim=1)
        x2 = self.conv3(x2)
        x2 = self.BN3(x2)
        x2 = self.ReLU(x2)

        # print(x2.shape)

        x1 = torch.cat((x1,x2),dim=1)
        # print(x1.shape)
        x1 = self.conv4(x1)
        x1 = self.BN4(x1)
        x1 = self.ReLU(x1)
        x1 = self.conv5(x1)
        x1 = self.BN5(x1)
        x1 = self.ReLU(x1)
        x = self.conv6(x1)


        return x



# 定义了一个网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.resnet = resnet18(pretrained=False)  # 调用 resnet.py 文件中的这个类
        self.Encoder = Encoder()
        self.Stem = StemLayer(in_channels=3, out_channels=16)
        self.changechannellayer = ChannelReductionBlock(3, 1).to('cuda:0')
        self.S_Decoder1 = SELF_decoder(start_channel=16)
        self.S_Decoder2 = SELF_decoder(start_channel=32)
        self.S_Decoder3 = SELF_decoder(start_channel=64)
        self.S_Decoder4 = SELF_decoder(start_channel=128)
        self.Total_Decoder1 = ToTal_decoder(start_channel=16)
        self.Total_Decoder2 = ToTal_decoder(start_channel=32)
        self.Total_Decoder3 = ToTal_decoder(start_channel=64)
        self.Total_Decoder4 = ToTal_decoder(start_channel=128)
        self.Rebuilt1 = Rebuilt(8, 4)
        self.Rebuilt2 = Rebuilt(16, 8)
        self.Rebuilt3 = Rebuilt(32, 16)
        self.Rebuilt4 = Rebuilt(64, 32)
        self.Generator = Generator(24,12,6,3)
        self.GetMap1 = GetMap(24)
        self.GetMap2 = GetMap(48)
        self.GetMap3 = GetMap(96)
        self.GetMap4 = GetMap(192)
        self.Map = GetChangeMap(6,12,24,48)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, x1, x2):
        # Stem层
        print(x1)
        print(x2)
        input1_tensor = self.Stem(x1)
        input2_tensor = self.Stem(x2)
        # print(input1_tensor.shape)
        # 编码器部分
        list1 = self.Encoder(input1_tensor)  # 编码器部分的输出： torch.Size([12, 32, 256, 256]) torch.Size([12, 64, 128, 128]) torch.Size([12, 128, 64, 64]) torch.Size([12, 256, 32, 32])
        list2 = self.Encoder(input2_tensor)

        #
        # list1 = self.resnet(x1)
        # list2 = self.resnet(x2)

        print(list1[0].shape,list1[1].shape,list1[2].shape,list1[3].shape)
        # torch.Size([1, 16, 256, 256]) torch.Size([1, 32, 128, 128]) torch.Size([1, 64, 64, 64]) torch.Size([1, 128, 32, 32])
        # G1_11表示 第一个图的 第一个尺度 的公有    G1_22表示第一个图的第二个尺度的私有
        G1_11 = self.S_Decoder1(list1[0])
        G1_21 = self.S_Decoder2(list1[1])
        G1_31 = self.S_Decoder3(list1[2])
        G1_41 = self.S_Decoder4(list1[3])

        G1_12 = self.Total_Decoder1(list1[0])
        G1_22 = self.Total_Decoder2(list1[1])
        G1_32 = self.Total_Decoder3(list1[2])
        G1_42 = self.Total_Decoder4(list1[3])

        G2_11 = self.S_Decoder1(list2[0])
        G2_21 = self.S_Decoder2(list2[1])
        G2_31 = self.S_Decoder3(list2[2])
        G2_41 = self.S_Decoder4(list2[3])

        G2_12 = self.Total_Decoder1(list2[0])
        G2_22 = self.Total_Decoder2(list2[1])
        G2_32 = self.Total_Decoder3(list2[2])
        G2_42 = self.Total_Decoder4(list2[3])

        # 第一个图的私有和第一个图的公有
        G11 = self.Rebuilt1(G1_12, G1_11)
        G12 = self.Rebuilt2(G1_22, G1_21)
        G13 = self.Rebuilt3(G1_32, G1_31)
        G14 = self.Rebuilt4(G1_42, G1_41)
        # 第一个图的私有和第二个图的公有
        G21 = self.Rebuilt1(G1_12, G2_11)
        G22 = self.Rebuilt2(G1_22, G2_21)
        G23 = self.Rebuilt3(G1_32, G2_31)
        G24 = self.Rebuilt4(G1_42, G2_41)
        # 第二个图的私有和第二个图的公有
        G31 = self.Rebuilt1(G2_12, G2_11)
        G32 = self.Rebuilt2(G2_22, G2_21)
        G33 = self.Rebuilt3(G2_32, G2_31)
        G34 = self.Rebuilt4(G2_42, G2_41)
        # 第一个图的公有和第二个图的私有
        G41 = self.Rebuilt1(G2_12, G1_11)
        G42 = self.Rebuilt2(G2_22, G1_21)
        G43 = self.Rebuilt3(G2_32, G1_31)
        G44 = self.Rebuilt4(G2_42, G1_41)

        G1 = self.Generator(G11, G12, G13, G14)
        G2 = self.Generator(G21, G22, G23, G24)
        G3 = self.Generator(G31, G32, G33, G34)
        G4 = self.Generator(G41, G42, G43, G44)

        map1 = self.GetMap1(G1_11, G2_11, G1_12, G2_12)
        map2 = self.GetMap2(G1_21, G2_21, G1_22, G2_22)
        map3 = self.GetMap3(G1_31, G2_31, G1_32, G2_32)
        map4 = self.GetMap4(G1_41, G2_41, G1_42, G2_42)



        Map = self.Map(map1, map2, map3, map4)

        return G1,G2,G3,G4,Map

