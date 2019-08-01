from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                        nn.BatchNorm2d(out_channel))

def conv3x3(in_channel, out_channel, stride = 1):
    return nn.Conv2d(in_channel, out_channel, kernal_size = 3, stride = stride, padding = 1, bias = False)

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                        nn.BatchNorm3d(out_channel))
class bottleneck_layer(nn.Module):
    expansion = 4#class attribute
    def __init__(self, inplanes, planes , stride = 1, downsample = None):
        super(bottleneck_layer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class image_pre_extraction(nn.Module):
    def __init__(self):
        super(image_pre_extraction, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class disp_pre_extraction(nn.Module):
    def __init__(self):
        super(disp_pre_extraction, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

class feature_extraction(nn.Module):
    def __init__(self, layers):
        self.inplanes = 64
        self.layers = layers
        block = bottleneck_layer
        super(feature_extraction, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, self.layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.layers[2], stride=2)
        if self.layers[3] > 0:
            self.layer4 = self._make_layer(block, 512, self.layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion#every time the output channel is input channel * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        if self.layers[3] > 0:
            x5 = self.layer4(x4)
        else:
            x5 = None
        return x, x1, x2, x3, x4, x5


class upconv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upconv,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = F.interpolate(x, scale_factor=2)
        x = self.up(x)
        return x

class get_disp(nn.Module):
    def __init__(self, ch_in):
        super(get_disp,self).__init__()
        self.get_disp_conv = nn.Sequential(
            nn.Conv2d(ch_in, 32, kernel_size=3, stride=1, padding=1,bias=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,bias=False),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1,bias=False)
        )
        
    def forward(self, x, max_disp):
        x = self.get_disp_conv(x)
        x = torch.clamp(x, -max_disp/2 , max_disp/2)
        return x

class get_disp_dilate(nn.Module):
    def __init__(self, ch_in):
        super(get_disp_dilate,self).__init__()
        self.get_disp_conv = nn.Sequential(
            nn.Conv2d(ch_in, 64, kernel_size=3, stride=1, padding=1,dilation=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation = 1, bias=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation = 1, bias=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, dilation = 1, bias=False),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, x, max_disp):
        x = self.get_disp_conv(x)
        x = torch.clamp(x, -max_disp/2 , max_disp/2)
        return x

def scale_pyramid(img, num_scale = 4):   
    scaled_imgs = [img]
    #print(shape)
    for i in range(num_scale - 1):
        scaled_imgs.append(F.interpolate(img,scale_factor=0.5**(i+1),mode='bilinear'))

    return scaled_imgs



class loss():
    def __init__(self, res_output, output, lr_pyramid, disp_gt, mask):
        self.res_output = res_output
        self.lr_pyramid = [lp.detach() for lp in lr_pyramid]
        self.disp_gt = disp_gt
        self.mask = mask
        self.output = output
        self.res_gt = [self.disp_gt[i] - self.lr_pyramid[i] for i in range(4)]
        self.res_gt = [g.detach() for g in self.res_gt]
        o_loss_list = [F.smooth_l1_loss(self.lr_pyramid[i]*self.mask[i], self.disp_gt[i]*self.mask[i], size_average=True) for i in range(4)]
        self.o_loss = sum(o_loss_list)
        self.res_loss()
        self.global_loss()
    
    @staticmethod
    def SSIM(x,y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)
    
    def res_loss(self):
        l1_res_loss_list = [F.smooth_l1_loss(self.res_output[i]*self.mask[i], self.res_gt[i]*self.mask[i], size_average=True) for i in range(4)]
        self.l1_res_loss = sum(l1_res_loss_list)
        
        self.diff_loss = torch.exp(self.l1_res_loss - self.o_loss)

        SSIM_loss_list = [torch.mean(self.SSIM(self.output[i] * self.mask[i], self.disp_gt[i] * self.mask[i])) for i in range(4)]
        self.SSIM_loss = sum(SSIM_loss_list)
        
        lr_l1_loss_list = [F.smooth_l1_loss(self.output[i]*self.mask[i], self.lr_pyramid[i]*self.mask[i],size_average=True) for i in range(4)]
        self.lr_l1_loss = -sum(lr_l1_loss_list)
        self.lr_l1_loss = torch.exp(self.lr_l1_loss)
        #return self.l1_res_loss, self.diff_loss, self.SSIM_loss, self.lr_l1_loss
    
    def global_loss(self):
        l1_full_loss_list = [F.smooth_l1_loss(self.output[i], self.disp_gt[i], size_average=True) for i in range(4)]
        self.l1_full_loss = sum(l1_full_loss_list)
        SSIM_full_loss_list = [torch.mean(self.SSIM(self.output[i], self.disp_gt[i])) for i in range(4)]
        self.SSIM_full_loss = sum(SSIM_full_loss_list)
        #return self.l1_full_loss, self.SSIM_full_loss