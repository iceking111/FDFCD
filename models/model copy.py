import torch
import torch.nn as nn
from torch.distributions import *
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from thop import profile
# 编码器结构  其中的dims中存储的就是四个不同尺度
# 编码器输出尺寸： torch.Size([1, 256, 256, 32]) torch.Size([1, 32, 256, 256]) torch.Size([1, 64, 128, 128]) torch.Size([1, 128, 64, 64]) torch.Size([1, 256, 32, 32])
# 首先把stem输出的结果改变一下，改为24   然后按照通道进行划分为  6   18
# 将B 6 256 256以及B 18 256 256分别放入编码器 最后得到四个尺度
def calculate_f1_score(Map, label):
    Map = Map.detach()
    Map = torch.argmax(Map, dim=1,keepdim=True) # 输出的形状是B 256 256
    # 确保Map和label是相同形状的单通道图像
    assert Map.shape == label.shape, "Map and label must have the same shape."
    
    # 计算真阳性（TP）、假阳性（FP）、真阴性（TN）和假阴性（FN）
    TP = torch.sum((Map == 1) & (label == 1))
    FP = torch.sum((Map == 1) & (label == 0))
    FN = torch.sum((Map == 0) & (label == 1))
    
    # 计算精确率（Precision）和召回率（Recall）
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score

# 准备部分类代码
class resblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(3,3), stride=(1,1),padding=1):
        super(resblock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding)
        self.IN = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x_res = self.conv(x)
        x_res = self.IN(x_res)
        x_res = self.relu(x_res)
        x_res = self.conv(x_res)
        x_res = self.IN(x_res)
        return x+x_res


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet1(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512,256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc5 = OutConv(512, 256)
        self.outc4 = OutConv(512, 256)
        self.outc3 = OutConv(256, 128)
        self.outc2 = OutConv(128, 64)
        self.outc1 = OutConv(64, 32)

    def forward(self, x ,y):
        # self.inc是将通道扩到32  其中有两个卷积层
        x1 = self.inc(x) # 32 256 256
        # self.down1-4是先进行一个最大池化操作将尺寸下降一倍  在经历一组卷积层
        x2 = self.down1(x1) # 64 128 128
        x3 = self.down2(x2) # 128 64 64
        x4 = self.down3(x3) # 256 32 32
        x5 = self.down4(x4) # 512 16 16
        # self.inc是将通道扩到32  其中有两个卷积层
        y1 = self.inc(y)
        # self.down1-4是先进行一个最大池化操作将尺寸下降一倍  在经历一组卷积层
        y2 = self.down1(y1) # 64 128 128
        y3 = self.down2(y2) # 128 64 64
        y4 = self.down3(y3) # 256 32 32
        y5 = self.down4(y4) # 512 16 16
        z5 = x5 - y5
        z4 = x4 - y4
        z3 = x3 - y3
        z2 = x2 - y2
        z1 = x1 - y1
        x = self.up1(z5, z4)
        x = self.up2(x, z3)
        x = self.up3(x, z2)
        x = self.up4(x, z1)
        return x

class UNet2(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512,256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc5 = OutConv(512, 256)
        self.outc4 = OutConv(512, 256)
        self.outc3 = OutConv(256, 128)
        self.outc2 = OutConv(128, 64)
        self.outc1 = OutConv(64, 32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

# 特征解耦模块
class Decoupling_module(nn.Module):
    def __init__(self):
        super(Decoupling_module, self).__init__()
        self.getT1C = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    resblock(in_channels=64,out_channels=64),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1)))
        self.getT2C = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    resblock(in_channels=64,out_channels=64),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1)))
        self.getTNC = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    resblock(in_channels=64,out_channels=64),
                                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(2,2),padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(1,1),stride=(1,1)))
    def forward(self,x):
        T1C = self.getT1C(x)
        T2C = self.getT2C(x)
        TNC = self.getTNC(x)
        return T1C,T2C,TNC

# 重构模块
class Reconstruct_module(nn.Module):
    def __init__(self):
        super(Reconstruct_module, self).__init__()
        self.reconpicture = nn.Sequential(nn.ConvTranspose2d(in_channels=192,out_channels=48,kernel_size=(3,3),stride=(2,2),padding=1,output_padding=1),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    # resblock(in_channels=64,out_channels=64),
                                    nn.ConvTranspose2d(in_channels=48,out_channels=12,kernel_size=(3,3),stride=(2,2),padding=1,output_padding=1),
                                    nn.BatchNorm2d(12),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    resblock(in_channels=12,out_channels=12),
                                    # nn.Conv2d(in_channels=48,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=1),
                                    # nn.BatchNorm2d(8),
                                    # nn.ReLU(),
                                    nn.Conv2d(in_channels=12,out_channels=3,kernel_size=(3,3),stride=(1,1),padding=1),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(),
                                    # nn.Dropout2d(0.1),
                                    nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(1,1),stride=(1,1)),
                                    nn.Tanh(),
                                    )
    def recon(self,TC,TNC):
        input = torch.cat((TC,TNC),dim=1)
        out = self.reconpicture(input)
        return out

    def forward(self,T1C,T2C,TNC):
        recon_image1 = self.recon(T1C,TNC)
        recon_image2 = self.recon(T2C,TNC)
        return recon_image1,recon_image2

# 生成分布的模块



class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, inchannels,num_filters, latent_dim, num_output_channels,no_convs_fcomb, initializers, use_tile=True):
        '''
        :param num_filters:就是卷积中的通道数
        :param latent_dim:潜在空间的维度 
        :param num_output_channels:
        :param num_classes: 最后想要分为几类  也就是最后输出的特征的通道数
        :param no_convs_fcomb:表示在模块中除了最后一层卷积层之外，中间所含的1×1卷积层的数量
        :param initializers: 使用什么初始化方法
        :param use_tile: 决定是否采用tile相关的操作逻辑
        '''
        super(Fcomb, self).__init__()
        self.inchannels = inchannels
        self.num_channels = num_output_channels #output channels
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'
        self.device = 'cuda:0'
        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(inchannels, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[_], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_channels, kernel_size=1)

            # if initializers['w'] == 'orthogonal':
            #     self.layers.apply(init_weights_orthogonal_normal)
            #     self.last_layer.apply(init_weights_orthogonal_normal)
            # else:
            self.layers.apply(init_weights)
            self.last_layer.apply(init_weights)
        self.activation = torch.nn.Softmax(dim=1)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(self.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, use_softmax=False):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            output = self.last_layer(output)
            if use_softmax:
                output = self.activation(output)

            # print(output.shape)
            return output

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


class KLEncoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        '''
        :param input_channels: 这个参数是输入通道数
        :param num_filters: 这个参数是卷积核数量列表
        :param no_convs_per_block: 这个参数是每个模块中的卷积层数
        :param initializers: 初始化方法
        :param padding: 是否填充
        :param posterior: 是否是后验
        '''
        super(KLEncoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 4

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

# 这个是生成先验分布的结构
class Prior_Gaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        '''
        :param input_channels: 这个参数是输入的通道数
        :param num_filters: 这个参数是卷积核数量列表
        :param no_convs_per_block: 这个参数是每个模块中的卷积层数
        :param latent_dim: 这个参数是潜在空间维度
        :param initializers: 初始化方法
        :param posterior:是否是后验
        '''
        super(Prior_Gaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = KLEncoder(input_channels=self.input_channels, num_filters=self.num_filters, no_convs_per_block=self.no_convs_per_block, initializers=initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            # segm = torch.unsqueeze(segm, 1)
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding
        # print("经过KLEncoder之后的",encoding.shape)
        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        
        # print("全局平均池化之后的结果",encoding.shape)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)
        # print("卷积之后的结果",mu_log_sigma.shape)
        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        # print("第一次分离之后的结果",mu_log_sigma.shape)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        # print("第二次分离之后的结果",mu_log_sigma.shape)
        # print(mu_log_sigma.shape)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist


#  这个是生成后验分布的结构
class Post_Gaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        '''
        :param input_channels: 这个参数是输入的通道数
        :param num_filters: 这个参数是卷积核数量列表
        :param no_convs_per_block: 这个参数是每个模块中的卷积层数
        :param latent_dim: 这个参数是潜在空间维度
        :param initializers: 初始化方法
        :param posterior:是否是后验
        '''
        super(Post_Gaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        # self.encoder = KLEncoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(32, 12, (1,1), stride=1)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(input, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist


# 网络结构
# 定义了一个网络模型
class Net(nn.Module):
    def __init__(self, num_filters=[16,32,64],latent_dim=6, no_convs_fcomb=2):
        '''
        :param num_filters:
        :param latent_dim:
        :param no_convs_fcomb:
        '''
        super(Net, self).__init__()
        self.num_filters = num_filters
        self.no_convs_fcomb = no_convs_fcomb
        self.device = 'cuda:0'
        self.to(self.device)
        self.UNet1 = UNet1(3).to(self.device) # 用于生成解耦特征
        self.UNet2 = UNet2(3).to(self.device) # 用于生成先验分布以及后验分布等等
        self.decoupling_module  = Decoupling_module().to(self.device)# 解耦结构
        self.reconstruct_module = Reconstruct_module().to(self.device)# 重构模块
        self.latent_dim = latent_dim
        # 最开始生成先验变化分布
        self.change_prior = Prior_Gaussian(input_channels=3, num_filters=[16, 32, 64],
                                                          no_convs_per_block=3, latent_dim=6,
                                                          initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        # 最开始生成先验不变分布
        self.nonchange_prior = Prior_Gaussian(input_channels=3, num_filters=[16, 32, 64],
                                                               no_convs_per_block=3, latent_dim=6,
                                                               initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        # 在后续更新分布阶段  更新先验变化分布增量所采用的编码器
        self.prior_new_change_stage1 = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
                                                          no_convs_per_block=3, latent_dim=6,
                                                          initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        self.prior_new_unchange_stage1 = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
                                                    no_convs_per_block=3, latent_dim=6,
                                                    initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        self.prior_new_change_stage2 = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
                                                          no_convs_per_block=3, latent_dim=6,
                                                          initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        self.prior_new_unchange_stage2 = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
                                                    no_convs_per_block=3, latent_dim=6,
                                                    initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        self.prior_new_change_stage3 = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
                                                    no_convs_per_block=3, latent_dim=6,
                                                          initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        self.prior_new_unchange_stage3 = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
                                                    no_convs_per_block=3, latent_dim=6,
                                                    initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)                                                   
        # # 在后续更新分布阶段  更新先验不变分布增量所采用的编码器
        # self.nonchange_prior_new = Prior_Gaussian(input_channels=32, num_filters=[16, 32, 64],
        #                                                        no_convs_per_block=3, latent_dim=6,
        #                                                        initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        # 生成后验分布的结构   无论在最开始  还是在中间更新阶段 都是不变的
        self.post_change = Post_Gaussian(input_channels=32,
                                                                     num_filters=[16, 32, 64],
                                                                     no_convs_per_block=3, latent_dim=6,
                                                                     initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        self.post_unchange = Post_Gaussian(input_channels=32,
                                                                     num_filters=[16, 32, 64],
                                                                     no_convs_per_block=3, latent_dim=6,
                                                                     initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        
        # self.nonchange_post = Post_Gaussian(input_channels=32,
        #                                                             num_filters=[16, 32, 64],
        #                                                             no_convs_per_block=3, latent_dim=6,
        #                                                             initializers={'w': 'he_normal', 'b': 'normal'}).to(self.device)
        # 
        self.fcomb_change_1 = Fcomb(134, self.num_filters, self.latent_dim,1,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)
        self.fcomb_change_2 = Fcomb(134, self.num_filters, self.latent_dim,1,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)
        self.fcomb_change_3 = Fcomb(134, self.num_filters, self.latent_dim,1,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)
        self.fcomb_unchange_1 = Fcomb(134, self.num_filters, self.latent_dim,1,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)
        self.fcomb_unchange_2 = Fcomb(134, self.num_filters, self.latent_dim,1,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)
        self.fcomb_unchange_3 = Fcomb(134, self.num_filters, self.latent_dim,1,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(self.device)

        self.change_head_stage1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            # nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))
        self.unchange_head_stage1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))
        self.change_head_stage2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            # nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))
        self.unchange_head_stage2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))
        self.change_head_stage3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            # nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))
        self.unchange_head_stage3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            # nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))

    # 这是最开始产生的变化分布采样的过程
    def prior_sampling_change_stage1(self,feature,prior_distribution_stage,sample_num=20, training=True):
        samples = []
        for _ in range(sample_num):
            if training:
                z_change_prior = prior_distribution_stage.rsample()
            else:
                z_change_prior = prior_distribution_stage.sample()
            # print(z_change_prior.shape)
            sample = self.fcomb_change_1.forward(feature, z_change_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def prior_sampling_change_stage2(self,feature,prior_distribution_stage,sample_num=20, training=True):
        samples = []
        for _ in range(sample_num):
            if training:
                z_change_prior = prior_distribution_stage.rsample()
            else:
                z_change_prior = prior_distribution_stage.sample()
            # print(z_change_prior.shape)
            sample = self.fcomb_change_2.forward(feature, z_change_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def prior_sampling_change_stage3(self,feature,prior_distribution_stage,sample_num=20, training=True):
        samples = []
        for _ in range(sample_num):
            if training:
                z_change_prior = prior_distribution_stage.rsample()
            else:
                z_change_prior = prior_distribution_stage.sample()
            # print(z_change_prior.shape)
            sample = self.fcomb_change_3.forward(feature, z_change_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def prior_sampling_unchange_stage1(self,feature,prior_distribution_stage,sample_num=20, training=True):
        samples = []
        for _ in range(sample_num):
            if training:
                z_change_prior = prior_distribution_stage.rsample()
            else:
                z_change_prior = prior_distribution_stage.sample()
            # print(z_change_prior.shape)
            sample = self.fcomb_unchange_1.forward(feature, z_change_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def prior_sampling_unchange_stage2(self,feature,prior_distribution_stage,sample_num=20, training=True):
        samples = []
        for _ in range(sample_num):
            if training:
                z_change_prior = prior_distribution_stage.rsample()
            else:
                z_change_prior = prior_distribution_stage.sample()
            # print(z_change_prior.shape)
            sample = self.fcomb_unchange_2.forward(feature, z_change_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def prior_sampling_unchange_stage3(self,feature,prior_distribution_stage,sample_num=20, training=True):
        samples = []
        for _ in range(sample_num):
            if training:
                z_change_prior = prior_distribution_stage.rsample()
            else:
                z_change_prior = prior_distribution_stage.sample()
            # print(z_change_prior.shape)
            sample = self.fcomb_unchange_3.forward(feature, z_change_prior, use_softmax=False)
            samples.append(sample)
        return samples
    # # 这是最开始不变分布采样的过程
    # def prior_sampling_unchange(self,feature,prior_distribution_stage, sample_num=20, training=True):
    #     samples = []
    #     for _ in range(sample_num):
    #         if training:
    #             z_change_prior = prior_distribution_stage.rsample()
    #         else:
    #             z_change_prior = prior_distribution_stage.sample()
    #         # print(z_change_prior.shape)
    #         sample = self.fcomb1_unchange.forward(feature, z_change_prior, use_softmax=False)
    #         samples.append(sample)
    #     return samples
# 生成变化图的部分是训练和测试都是一致的
    def Get_Map_stage1(self,TC1,TC2,TNC,change_prior_distribution1,unchange_prior_distribution1,training):

        # 首先把两个变化特征级联，然后与先验变化分布融合,不变特征也类似
        change_feature = torch.cat((TC1,TC2),dim=1)    #   两个加起来通道数是128 
  
        unchange_feature = TNC    # 通道数是128
   
        # 对分布进行采样得到特征

        if training:
            prior_change_feature_list = self.prior_sampling_change_stage1(change_feature,change_prior_distribution1,32, training)
            prior_nonchange_feature_list = self.prior_sampling_unchange_stage1(unchange_feature,unchange_prior_distribution1,64, training)
        else:
            prior_change_feature_list = self.prior_sampling_change_stage1(change_feature,change_prior_distribution1,32, training)  
            prior_nonchange_feature_list = self.prior_sampling_unchange_stage1(unchange_feature,unchange_prior_distribution1,64, training)

        unchange_feature = torch.cat(prior_nonchange_feature_list, dim=1)  
        change_feature = torch.cat(prior_change_feature_list, dim=1)  #  64 64 64
        # print(change_feature.shape)


        # 先重参数化采样出不变特征

        # print(unchange_feature.shape)
        # 不变的特征与先验不变分布特征级联
        # unchange_feature = torch.cat((unchange_feature,prior_nonchange_feature),dim=1)  # 196

        # 变化头部分，直到生成变化图（一堆卷积卷到一通道进行级联）
        change_feature = self.change_head_stage1(change_feature)
        # 不变头部分，直到生成不变图
        unchange_feature = self.unchange_head_stage1(unchange_feature)

        Map1 = torch.cat((change_feature,unchange_feature),dim=1)

        # 更新分布
        return  Map1

    def Get_Map_stage2(self,TC1,TC2,TNC,change_prior_distribution1,unchange_prior_distribution1,training):

        # 首先把两个变化特征级联，然后与先验变化分布融合,不变特征也类似
        change_feature = torch.cat((TC1,TC2),dim=1)    #   两个加起来通道数是128 
  
        unchange_feature = TNC    # 通道数是128
   
        # 对分布进行采样得到特征

        if training:
            prior_change_feature_list = self.prior_sampling_change_stage2(change_feature,change_prior_distribution1,32,training)
            prior_nonchange_feature_list = self.prior_sampling_unchange_stage2(unchange_feature,unchange_prior_distribution1,64,training)
        else:
            prior_change_feature_list = self.prior_sampling_change_stage2(change_feature,change_prior_distribution1,32,training)  
            prior_nonchange_feature_list = self.prior_sampling_unchange_stage2(unchange_feature,unchange_prior_distribution1,64, training)

        unchange_feature = torch.cat(prior_nonchange_feature_list, dim=1)
        change_feature = torch.cat(prior_change_feature_list, dim=1)  #  64 64 64
        # print(change_feature.shape)


        # 先重参数化采样出不变特征

        # print(unchange_feature.shape)
        # 不变的特征与先验不变分布特征级联
        # unchange_feature = torch.cat((unchange_feature,prior_nonchange_feature),dim=1)  # 196

        # 变化头部分，直到生成变化图（一堆卷积卷到一通道进行级联）
        change_feature = self.change_head_stage2(change_feature)
        # 不变头部分，直到生成不变图
        unchange_feature = self.unchange_head_stage2(unchange_feature)

        Map2 = torch.cat((change_feature,unchange_feature),dim=1)

        # 更新分布
        return  Map2

    def Get_Map_stage3(self,TC1,TC2,TNC,change_prior_distribution1,unchange_prior_distribution1,training):

        # 首先把两个变化特征级联，然后与先验变化分布融合,不变特征也类似
        change_feature = torch.cat((TC1,TC2),dim=1)    #   两个加起来通道数是128 
  
        unchange_feature = TNC    # 通道数是128
   
        # 对分布进行采样得到特征
        if training:
            prior_change_feature_list = self.prior_sampling_change_stage3(change_feature,change_prior_distribution1,32, training)
            prior_nonchange_feature_list = self.prior_sampling_unchange_stage3(unchange_feature,unchange_prior_distribution1,64, training)
        else:
            prior_change_feature_list = self.prior_sampling_change_stage3(change_feature,change_prior_distribution1,32,training)  
            prior_nonchange_feature_list = self.prior_sampling_unchange_stage3(unchange_feature,unchange_prior_distribution1,64,training)
        unchange_feature = torch.cat(prior_nonchange_feature_list, dim=1)
        change_feature = torch.cat(prior_change_feature_list, dim=1)  #  64 64 64
        # print(change_feature.shape)


        # 先重参数化采样出不变特征

        # print(unchange_feature.shape)
        # 不变的特征与先验不变分布特征级联
        # unchange_feature = torch.cat((unchange_feature,prior_nonchange_feature),dim=1)  # 196

        # 变化头部分，直到生成变化图（一堆卷积卷到一通道进行级联）
        change_feature = self.change_head_stage3(change_feature)
        # 不变头部分，直到生成不变图
        unchange_feature = self.unchange_head_stage3(unchange_feature)

        Map3 = torch.cat((change_feature,unchange_feature),dim=1)

        # 更新分布
        return  Map3

    # 训练阶段的更新分布
    def Get_Distribution_train_stage1(self,Map,feature,change_prior_distribution_old,unchange_prior_distribution_old,label):

        # 首先把Map和lable叠一下  得到绿色和红色的图
        # 先把Map化为和lable一样的单通道
        Map = Map.detach()
        Map = torch.argmax(Map, dim=1, keepdim=True)
        # 0是黑色（不变的）  1是白色（变化的）
        # Map是黑色的 但是label是白色的  此时应该是绿色    
        output_green = torch.zeros((Map.shape[0], 1, 256, 256), dtype=torch.uint8)
        # 找到满足条件的像素位置
        condition = (Map == 0) & (label == 1)

        # 将满足条件的像素位置设置为红色   但是实际上还是一个单通道的黑白图  只要红色部分值是1就行 为了方便后续乘
        output_green[:, 0, :, :][condition.squeeze(1)] = 1
        output_green = output_green.to('cuda:0')
        # output_green = torch.where(condition, torch.tensor(1, dtype=torch.uint8), output_green)

        # Map是白色的  但是label是黑色的  此时应该是红色
        output_red = torch.zeros((Map.shape[0], 1, 256, 256), dtype=torch.uint8)
        # 找到满足条件的像素位置
        condition = (Map == 1) & (label == 0)

        # 将满足条件的像素位置设置为红色
        # output_red = torch.where(condition, torch.tensor(1, dtype=torch.uint8), output_red)
        output_red[:, 0, :, :][condition.squeeze(1)] = 1
        output_red = output_red.to('cuda:0')
        # 分别把红绿和unet出来的特征相乘得到这一阶段未检测到的变化特征和不变特征
        change = feature  * output_green   # 32 256 256 
        unchange = feature * output_red    # 32 256 256
        # 分别计算均值方差来得到后验分布
        Change_post_distribution = self.post_change.forward(change)
        unchange_post_distribution = self.post_unchange.forward(unchange)
        # 只使用unet出来的特征输入两个独立的编码器来得到两个先验分布增量
        change_prior_distribution_delta = self.prior_new_change_stage1.forward(feature)
        unchange_prior_distribution_delta = self.prior_new_unchange_stage1.forward(feature)
        # 计算所得的增量分布与后验分布的kl散度
        kl_loss = torch.mean(kl_divergence(change_prior_distribution_delta,Change_post_distribution) + kl_divergence(unchange_prior_distribution_delta,unchange_post_distribution), dim=0)
        # 更新分布  把先验分布增量作用于前一个阶段的分布上 得到更新之后的分布
        change_prior_mean_old = change_prior_distribution_old.mean
        change_prior_std_old = change_prior_distribution_old.stddev
        change_prior_mean_delta = change_prior_distribution_delta.mean
        change_prior_std_delta = change_prior_distribution_delta.stddev

        unchange_prior_mean_old = unchange_prior_distribution_old.mean
        unchange_prior_std_old = unchange_prior_distribution_old.stddev
        unchange_prior_mean_delta = unchange_prior_distribution_delta.mean
        unchange_prior_std_delta = unchange_prior_distribution_delta.stddev

        change_prior_mean_new = change_prior_mean_old + change_prior_mean_delta
        change_prior_std_new = change_prior_std_old + change_prior_std_delta
        unchange_prior_mean_new = unchange_prior_mean_old + unchange_prior_mean_delta
        unchange_prior_std_new = unchange_prior_std_old + unchange_prior_std_delta


        change_prior_distribution_new = Independent(Normal(loc=change_prior_mean_new, scale=change_prior_std_new), 1)
        unchange_prior_distribution_new = Independent(Normal(loc=unchange_prior_mean_new, scale=unchange_prior_std_new), 1)
        return  change_prior_distribution_new,unchange_prior_distribution_new,kl_loss

    def Get_Distribution_train_stage2(self,Map,feature,change_prior_distribution_old,unchange_prior_distribution_old,label):

        # 首先把Map和lable叠一下  得到绿色和红色的图
        # 先把Map化为和lable一样的单通道
        Map = Map.detach()
        Map = torch.argmax(Map, dim=1, keepdim=True)
        # 0是黑色（不变的）  1是白色（变化的）
        # Map是黑色的 但是label是白色的  此时应该是绿色    
        output_green = torch.zeros((Map.shape[0], 1, 256, 256), dtype=torch.uint8)
        # 找到满足条件的像素位置
        condition = (Map == 0) & (label == 1)

        # 将满足条件的像素位置设置为红色   但是实际上还是一个单通道的黑白图  只要红色部分值是1就行 为了方便后续乘
        output_green[:, 0, :, :][condition.squeeze(1)] = 1
        output_green = output_green.to('cuda:0')
        # output_green = torch.where(condition, torch.tensor(1, dtype=torch.uint8), output_green)

        # Map是白色的  但是label是黑色的  此时应该是红色
        output_red = torch.zeros((Map.shape[0], 1, 256, 256), dtype=torch.uint8)
        # 找到满足条件的像素位置
        condition = (Map == 1) & (label == 0)

        # 将满足条件的像素位置设置为红色
        # output_red = torch.where(condition, torch.tensor(1, dtype=torch.uint8), output_red)
        output_red[:, 0, :, :][condition.squeeze(1)] = 1
        output_red = output_red.to('cuda:0')
        # 分别把红绿和unet出来的特征相乘得到这一阶段未检测到的变化特征和不变特征
        change = feature  * output_green   # 32 256 256 
        unchange = feature * output_red    # 32 256 256
        # 分别计算均值方差来得到后验分布
        Change_post_distribution = self.post_change.forward(change)
        unchange_post_distribution = self.post_unchange.forward(unchange)
        # 只使用unet出来的特征输入两个独立的编码器来得到两个先验分布增量
        change_prior_distribution_delta = self.prior_new_change_stage2.forward(feature)
        unchange_prior_distribution_delta = self.prior_new_unchange_stage2.forward(feature)
        # 计算所得的增量分布与后验分布的kl散度
        kl_loss = torch.mean(kl_divergence(change_prior_distribution_delta,Change_post_distribution) + kl_divergence(unchange_prior_distribution_delta,unchange_post_distribution), dim=0)
        # 更新分布  把先验分布增量作用于前一个阶段的分布上 得到更新之后的分布
        change_prior_mean_old = change_prior_distribution_old.mean
        change_prior_std_old = change_prior_distribution_old.stddev
        change_prior_mean_delta = change_prior_distribution_delta.mean
        change_prior_std_delta = change_prior_distribution_delta.stddev

        unchange_prior_mean_old = unchange_prior_distribution_old.mean
        unchange_prior_std_old = unchange_prior_distribution_old.stddev
        unchange_prior_mean_delta = unchange_prior_distribution_delta.mean
        unchange_prior_std_delta = unchange_prior_distribution_delta.stddev

        change_prior_mean_new = change_prior_mean_old + change_prior_mean_delta
        change_prior_std_new = change_prior_std_old + change_prior_std_delta
        unchange_prior_mean_new = unchange_prior_mean_old + unchange_prior_mean_delta
        unchange_prior_std_new = unchange_prior_std_old + unchange_prior_std_delta


        change_prior_distribution_new = Independent(Normal(loc=change_prior_mean_new, scale=change_prior_std_new), 1)
        unchange_prior_distribution_new = Independent(Normal(loc=unchange_prior_mean_new, scale=unchange_prior_std_new), 1)
        return  change_prior_distribution_new,unchange_prior_distribution_new,kl_loss

    def Get_Distribution_train_stage3(self,Map,feature,change_prior_distribution_old,unchange_prior_distribution_old,label):

        # 首先把Map和lable叠一下  得到绿色和红色的图
        # 先把Map化为和lable一样的单通道
        Map = Map.detach()
        Map = torch.argmax(Map, dim=1, keepdim=True)
        # 0是黑色（不变的）  1是白色（变化的）
        # Map是黑色的 但是label是白色的  此时应该是绿色    
        output_green = torch.zeros((Map.shape[0], 1, 256, 256), dtype=torch.uint8)
        # 找到满足条件的像素位置
        condition = (Map == 0) & (label == 1)

        # 将满足条件的像素位置设置为红色   但是实际上还是一个单通道的黑白图  只要红色部分值是1就行 为了方便后续乘
        output_green[:, 0, :, :][condition.squeeze(1)] = 1
        output_green = output_green.to('cuda:0')
        # output_green = torch.where(condition, torch.tensor(1, dtype=torch.uint8), output_green)

        # Map是白色的  但是label是黑色的  此时应该是红色
        output_red = torch.zeros((Map.shape[0], 1, 256, 256), dtype=torch.uint8)
        # 找到满足条件的像素位置
        condition = (Map == 1) & (label == 0)

        # 将满足条件的像素位置设置为红色
        # output_red = torch.where(condition, torch.tensor(1, dtype=torch.uint8), output_red)
        output_red[:, 0, :, :][condition.squeeze(1)] = 1
        output_red = output_red.to('cuda:0')
        # 分别把红绿和unet出来的特征相乘得到这一阶段未检测到的变化特征和不变特征
        change = feature  * output_green   # 32 256 256 
        unchange = feature * output_red    # 32 256 256
        # 分别计算均值方差来得到后验分布
        Change_post_distribution = self.post_change.forward(change)
        unchange_post_distribution = self.post_unchange.forward(unchange)
        # 只使用unet出来的特征输入两个独立的编码器来得到两个先验分布增量
        change_prior_distribution_delta = self.prior_new_change_stage3.forward(feature)
        unchange_prior_distribution_delta = self.prior_new_unchange_stage3.forward(feature)
        # 计算所得的增量分布与后验分布的kl散度
        kl_loss = torch.mean(kl_divergence(change_prior_distribution_delta,Change_post_distribution) + kl_divergence(unchange_prior_distribution_delta,unchange_post_distribution), dim=0)
        # 更新分布  把先验分布增量作用于前一个阶段的分布上 得到更新之后的分布
        change_prior_mean_old = change_prior_distribution_old.mean
        change_prior_std_old = change_prior_distribution_old.stddev
        change_prior_mean_delta = change_prior_distribution_delta.mean
        change_prior_std_delta = change_prior_distribution_delta.stddev

        unchange_prior_mean_old = unchange_prior_distribution_old.mean
        unchange_prior_std_old = unchange_prior_distribution_old.stddev
        unchange_prior_mean_delta = unchange_prior_distribution_delta.mean
        unchange_prior_std_delta = unchange_prior_distribution_delta.stddev

        change_prior_mean_new = change_prior_mean_old + change_prior_mean_delta
        change_prior_std_new = change_prior_std_old + change_prior_std_delta
        unchange_prior_mean_new = unchange_prior_mean_old + unchange_prior_mean_delta
        unchange_prior_std_new = unchange_prior_std_old + unchange_prior_std_delta


        change_prior_distribution_new = Independent(Normal(loc=change_prior_mean_new, scale=change_prior_std_new), 1)
        unchange_prior_distribution_new = Independent(Normal(loc=unchange_prior_mean_new, scale=unchange_prior_std_new), 1)
        return  change_prior_distribution_new,unchange_prior_distribution_new,kl_loss



    # 测试阶段的更新分布
    def Get_Distribution_test_stage1(self,feature,change_prior_distribution_old,unchange_prior_distribution_old):


        # 只使用unet出来的特征输入两个独立的编码器来得到两个先验分布增量
        change_prior_distribution_delta = self.prior_new_change_stage1.forward(feature)
        unchange_prior_distribution_delta = self.prior_new_unchange_stage1.forward(feature)
        # 计算所得的增量分布与后验分布的kl散度
        # kl_loss = torch.mean(kl_divergence(change_prior_distribution_delta,Change_post_distribution) + kl_divergence(unchange_prior_distribution_delta,unchange_post_distribution), dim=0)
        # 更新分布  把先验分布增量作用于前一个阶段的分布上 得到更新之后的分布
        
        change_prior_mean_old = change_prior_distribution_old.mean
        change_prior_std_old = change_prior_distribution_old.stddev
        change_prior_mean_delta = change_prior_distribution_delta.mean
        change_prior_std_delta = change_prior_distribution_delta.stddev

        unchange_prior_mean_old = unchange_prior_distribution_old.mean
        unchange_prior_std_old = unchange_prior_distribution_old.stddev
        unchange_prior_mean_delta = unchange_prior_distribution_delta.mean
        unchange_prior_std_delta = unchange_prior_distribution_delta.stddev

        change_prior_mean_new = change_prior_mean_old + change_prior_mean_delta
        change_prior_std_new = change_prior_std_old + change_prior_std_delta

        unchange_prior_mean_new = unchange_prior_mean_old + unchange_prior_mean_delta
        unchange_prior_std_new = unchange_prior_std_old + unchange_prior_std_delta
        change_prior_distribution_new = Independent(Normal(loc=change_prior_mean_new, scale=change_prior_std_new), 1)
        unchange_prior_distribution_new = Independent(Normal(loc=unchange_prior_mean_new, scale=unchange_prior_std_new), 1)

        return  change_prior_distribution_new,unchange_prior_distribution_new
        
    def Get_Distribution_test_stage2(self,feature,change_prior_distribution_old,unchange_prior_distribution_old):


        # 只使用unet出来的特征输入两个独立的编码器来得到两个先验分布增量
        change_prior_distribution_delta = self.prior_new_change_stage2.forward(feature)
        unchange_prior_distribution_delta = self.prior_new_unchange_stage2.forward(feature)
        # 计算所得的增量分布与后验分布的kl散度
        # kl_loss = torch.mean(kl_divergence(change_prior_distribution_delta,Change_post_distribution) + kl_divergence(unchange_prior_distribution_delta,unchange_post_distribution), dim=0)
        # 更新分布  把先验分布增量作用于前一个阶段的分布上 得到更新之后的分布
        
        change_prior_mean_old = change_prior_distribution_old.mean
        change_prior_std_old = change_prior_distribution_old.stddev
        change_prior_mean_delta = change_prior_distribution_delta.mean
        change_prior_std_delta = change_prior_distribution_delta.stddev

        unchange_prior_mean_old = unchange_prior_distribution_old.mean
        unchange_prior_std_old = unchange_prior_distribution_old.stddev
        unchange_prior_mean_delta = unchange_prior_distribution_delta.mean
        unchange_prior_std_delta = unchange_prior_distribution_delta.stddev

        change_prior_mean_new = change_prior_mean_old + change_prior_mean_delta
        change_prior_std_new = change_prior_std_old + change_prior_std_delta

        unchange_prior_mean_new = unchange_prior_mean_old + unchange_prior_mean_delta
        unchange_prior_std_new = unchange_prior_std_old + unchange_prior_std_delta
        change_prior_distribution_new = Independent(Normal(loc=change_prior_mean_new, scale=change_prior_std_new), 1)
        unchange_prior_distribution_new = Independent(Normal(loc=unchange_prior_mean_new, scale=unchange_prior_std_new), 1)

        return  change_prior_distribution_new,unchange_prior_distribution_new

    def Get_Distribution_test_stage3(self,feature,change_prior_distribution_old,unchange_prior_distribution_old):


        # 只使用unet出来的特征输入两个独立的编码器来得到两个先验分布增量
        change_prior_distribution_delta = self.prior_new_change_stage3.forward(feature)
        unchange_prior_distribution_delta = self.prior_new_unchange_stage3.forward(feature)
        # 计算所得的增量分布与后验分布的kl散度
        # kl_loss = torch.mean(kl_divergence(change_prior_distribution_delta,Change_post_distribution) + kl_divergence(unchange_prior_distribution_delta,unchange_post_distribution), dim=0)
        # 更新分布  把先验分布增量作用于前一个阶段的分布上 得到更新之后的分布
        
        change_prior_mean_old = change_prior_distribution_old.mean
        change_prior_std_old = change_prior_distribution_old.stddev
        change_prior_mean_delta = change_prior_distribution_delta.mean
        change_prior_std_delta = change_prior_distribution_delta.stddev

        unchange_prior_mean_old = unchange_prior_distribution_old.mean
        unchange_prior_std_old = unchange_prior_distribution_old.stddev
        unchange_prior_mean_delta = unchange_prior_distribution_delta.mean
        unchange_prior_std_delta = unchange_prior_distribution_delta.stddev

        change_prior_mean_new = change_prior_mean_old + change_prior_mean_delta
        change_prior_std_new = change_prior_std_old + change_prior_std_delta

        unchange_prior_mean_new = unchange_prior_mean_old + unchange_prior_mean_delta
        unchange_prior_std_new = unchange_prior_std_old + unchange_prior_std_delta
        change_prior_distribution_new = Independent(Normal(loc=change_prior_mean_new, scale=change_prior_std_new), 1)
        unchange_prior_distribution_new = Independent(Normal(loc=unchange_prior_mean_new, scale=unchange_prior_std_new), 1)

        return  change_prior_distribution_new,unchange_prior_distribution_new

    def forward(self, input1, input2,label=None, training=False):
        # UNet对差异特征进行处理
        input = input1 - input2
        # unet  用于解耦
        out1 = self.UNet1(input1,input2)
        # unet 用于生成分布
        out2 = self.UNet2(input)

        # 处理后的特征经过特征解耦模块 生成解耦特征
        self.TC1,self.TC2,self.TNC = self.decoupling_module(out1)
        # print('解耦特征的尺寸',self.TC1.shape,self.TC2.shape,self.TNC.shape)

        # 生成先验分布  
        self.change_prior_distribution =self.change_prior.forward(input)
        self.nonchange_prior_distribution =self.nonchange_prior.forward(input)
        

        if training == True:
            # 生成重构图来对解耦特征进行约束
            img1,img2 = self.reconstruct_module(self.TC1,self.TC2,self.TNC)

            # 生成后验分布
            change_post = out2 * label
            nonchange_post = out2 * (1 - label)

            # 双时图像相减，双时图像共同生成的特征与标签生成变化特征和不变特征生成后验分布（只有训练阶段）
            # 这两个后验分布只有最开始需要用到 后面的就不使用这两个后验分布做KL损失
            self.change_post_distribution = self.post_change.forward(change_post)
            self.nonchange_post_distribution = self.post_unchange.forward(nonchange_post)
            # 进行了三个stage
            # 得到第一个阶段的map
            stage1_map = self.Get_Map_stage1(self.TC1,self.TC2,self.TNC,self.change_prior_distribution,self.nonchange_prior_distribution,training=self.training)
            # 得到第一个阶段更新之后的分布增量以及分布
            stage1_prior_change_distribution,stage1_prior_unchange_distribution,kl_loss_stage1 = self.Get_Distribution_train_stage1(stage1_map,out2,self.change_prior_distribution,self.nonchange_prior_distribution,label)
            # 得到第二个阶段的map
            stage2_map = self.Get_Map_stage2(self.TC1,self.TC2,self.TNC,stage1_prior_change_distribution,stage1_prior_unchange_distribution,training=self.training)
            # 得到第二个阶段更新之后的分布增量以及分布
            stage2_prior_change_distribution,stage2_prior_unchange_distribution,kl_loss_stage2 = self.Get_Distribution_train_stage2(stage2_map,out2,stage1_prior_change_distribution,stage1_prior_unchange_distribution,label)
            # 得到第三个阶段的map
            stage3_map = self.Get_Map_stage3(self.TC1,self.TC2,self.TNC,stage2_prior_change_distribution,stage2_prior_unchange_distribution,training=self.training)
            # 得到第三个阶段更新之后的分布增量以及分布
            # stage3_prior_change_distribution,stage3_prior_unchange_distribution,kl_loss_stage3 = self.Get_Distribution_train(stage3_map,out2,stage2_prior_change_distribution,stage2_prior_unchange_distribution,label)
            # 得到需要与分布增量比较的后验分布

            # # 生成每个阶段的变化图
            # 计算KL散度
            kl_loss0 = torch.mean(kl_divergence(self.change_prior_distribution,self.change_post_distribution) + kl_divergence(self.nonchange_prior_distribution,self.nonchange_post_distribution), dim=0)
            kl_loss1 = kl_loss_stage1
            kl_loss2 = kl_loss_stage2
            # kl_loss3 = kl_loss_stage3

            kl_loss = kl_loss0 + kl_loss1 + kl_loss2

            return kl_loss,img1,img2,stage1_map,stage2_map,stage3_map

        else:
            # 进行了三个stage
            # 得到第一个阶段的map
            stage1_map = self.Get_Map_stage1(self.TC1,self.TC2,self.TNC,self.change_prior_distribution,self.nonchange_prior_distribution,training=self.training)
            # 得到第一个阶段更新之后的分布增量以及分布
            stage1_prior_change_distribution,stage1_prior_unchange_distribution = self.Get_Distribution_test_stage1(out2,self.change_prior_distribution,self.nonchange_prior_distribution)
            # 得到第二个阶段的map
            stage2_map = self.Get_Map_stage2(self.TC1,self.TC2,self.TNC,stage1_prior_change_distribution,stage1_prior_unchange_distribution,training=self.training)
            # 得到第二个阶段更新之后的分布增量以及分布
            stage2_prior_change_distribution,stage2_prior_unchange_distribution = self.Get_Distribution_test_stage2(out2,stage1_prior_change_distribution,stage1_prior_unchange_distribution)
            # 得到第三个阶段的map
            stage3_map = self.Get_Map_stage3(self.TC1,self.TC2,self.TNC,stage2_prior_change_distribution,stage2_prior_unchange_distribution,training=self.training)
            # 得到第三个阶段更新之后的分布增量以及分布
            # stage3_prior_change_distribution,stage3_prior_unchange_distribution = self.Get_Distribution_test(out2,stage2_prior_change_distribution,stage2_prior_unchange_distribution)
            
            return stage1_map,stage2_map,stage3_map


if __name__ == '__main__':
    net = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    batch = 1
    x = torch.randn(batch, 3, 256, 256)
    x = x.to(device)
    y = torch.randn(batch, 3, 256, 256)
    y = y.to(device)
    # 生产一个假定的标签 通道数为2 里面只有0和1
    mask = torch.ones(batch, 1, 256, 256)
    mask[:, 0, :, :] = torch.randint(0, 2, (256, 256))  # 黑色和白色
    mask = mask.to(device)
    # print(mask)
    model(x,y)
    # model(x,y)
    print("完美运行成功")

    macs, params = profile(model, inputs=(x, y))
    print(f"模型FLOPs: {macs/1e9:.2f} GFLOPs")
    print(f"模型参数量: {params/1e6:.2f} M params")
    

