from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
#################
#feature5 = 2048 H/64 too deep and useless
#feature4 = 1024 H/32
#feature3 = 512 H/16
#feature2 = 256 H/8
#feature1 = 64 H/4
#feature0 = 64 H/2
#################
class DSRnet(nn.Module):
    
    def __init__(self, max_disp, stage):
        super(DSRnet, self).__init__()
        self.stage = stage
        self.max_disp = max_disp
        #self.train_stage = train_stage
        layer = [2,4,3,0]
        self.feature_extraction = feature_extraction(layer)
        self.disp_pre_extraction = disp_pre_extraction()
        self.image_pre_extraction = image_pre_extraction()
        
        self.up5 = upconv(1024 * 2, 512)#H/16
        
        self.up4 = upconv(512 + 512 * 2, 256)#H/8
        self.disp4 = get_disp_dilate(256)
        
        self.up3 = upconv(256 + 256 * 2, 128)#H/4
        self.disp3 = get_disp_dilate(128+1)
        
        self.up2 = upconv(128 + 64 * 2,64)#H/2
        self.disp2 = get_disp_dilate(64+1)
        
        self.up1 = upconv(64 + 64*2, 64)#H
        self.disp1 = get_disp_dilate(64+1)

        if self.stage == 'distill':
            self.feature_extraction2 = feature_extraction(layer)
            self.up2_5 = upconv(1024 * 2, 512)#H/16
            
            self.up2_4 = upconv(512 + 512 * 2, 256)#H/8
            self.disp2_4 = get_disp_dilate(256+1)
            
            self.up2_3 = upconv(256 + 256 * 2, 128)#H/4
            self.disp2_3 = get_disp_dilate(128+1+1)
            
            self.up2_2 = upconv(128 + 64 * 2,64)#H/2
            self.disp2_2 = get_disp_dilate(64+1+1)
            
            self.up2_1 = upconv(64 + 64*2, 64)#H
            self.disp2_1 = get_disp_dilate(64+1+1)

    def forward(self,image,lr):
        # print(image.size())
        # print(lr.size())
        lr_pyramid = scale_pyramid(lr)
        #print(lr_pyramid[3].size())
        image_pre_feature = self.image_pre_extraction(image)
        disp_pre_feature = self.disp_pre_extraction(lr)
        
        image_feature     = self.feature_extraction(image_pre_feature)
        disp_feature  = self.feature_extraction(disp_pre_feature)
        # print(disp_feature[4].size()) 1024
        # print(disp_feature[3].size()) 512 
        # print(disp_feature[2].size()) 256 
        # print(disp_feature[1].size()) 64
        # print(disp_feature[0].size()) 64

        iconv5 = self.up5(torch.cat((image_feature[4], disp_feature[4]),1))#H/16
        #print(iconv5.size())
        iconv4 = self.up4(torch.cat((image_feature[3], disp_feature[3], iconv5), 1))#H/8
        #print(iconv4.size())
        output_disp1_4 = self.disp4(iconv4, self.max_disp)# + lr_pyramid[3]
        #print(output_disp4.size())
        up_disp4 = F.interpolate(output_disp1_4,scale_factor=2,mode='bilinear',align_corners=True)
        #print(up_disp4.size())
        iconv3 = self.up3(torch.cat((image_feature[2], disp_feature[2], iconv4), 1))#H/4
        #print(iconv3.size())
        output_disp1_3 = self.disp3(torch.cat((iconv3,up_disp4),1), self.max_disp)# + lr_pyramid[2]
        up_disp3 = F.interpolate(output_disp1_3,scale_factor=2,mode='bilinear',align_corners=True)

        iconv2 = self.up2(torch.cat((image_feature[1], disp_feature[1], iconv3), 1))#H/2
        output_disp1_2 = self.disp2(torch.cat((iconv2,up_disp3),1), self.max_disp)# + lr_pyramid[1]
        up_disp2 = F.interpolate(output_disp1_2,scale_factor=2,mode='bilinear',align_corners=True)
        
        iconv1 = self.up1(torch.cat((image_feature[0], disp_feature[0],iconv2),1))#H        
        output_disp1_1 = self.disp1(torch.cat((iconv1,up_disp2),1), self.max_disp)# + lr_pyramid[0]

        ###################stage two#######################
        if self.stage == 'distill':
            res_pre_feature = self.disp_pre_extraction(output_disp1_1)
            res_feature = self.feature_extraction2(res_pre_feature)

            iconv2_5 = self.up2_5(torch.cat((image_feature[4] , disp_feature[4] + res_feature[4]),1))
            
            iconv2_4 = self.up2_4(torch.cat((image_feature[3], disp_feature[3] + res_feature[3], iconv2_5), 1))#H/8
            output_disp2_4 = self.disp2_4( torch.cat((iconv2_4, output_disp1_4), 1), self.max_disp)
            up_disp2_4 = F.interpolate(output_disp2_4,scale_factor=2,mode='bilinear',align_corners=True)
            
            iconv2_3 = self.up2_3(torch.cat((image_feature[2], disp_feature[2] + res_feature[2], iconv2_4), 1))#H/4
            output_disp2_3 = self.disp2_3(torch.cat((iconv2_3,up_disp2_4,output_disp1_3),1), self.max_disp)
            up_disp2_3 = F.interpolate(output_disp2_3,scale_factor=2,mode='bilinear',align_corners=True)

            iconv2_2 = self.up2_2(torch.cat((image_feature[1], disp_feature[1] + res_feature[1], iconv2_3), 1))#H/2
            output_disp2_2 = self.disp2_2(torch.cat((iconv2_2,up_disp2_3,output_disp1_2),1), self.max_disp) 
            up_disp2_2 = F.interpolate(output_disp2_2,scale_factor=2,mode='bilinear',align_corners=True)
            
            iconv2_1 = self.up2_1(torch.cat((image_feature[0], disp_feature[0] + res_feature[0],iconv2_2),1))#H        
            output_disp2_1 = self.disp2_1(torch.cat((iconv2_1,up_disp2_2 ,output_disp1_1),1), self.max_disp)
        else:
            output_disp2_1 = output_disp2_2 = output_disp2_3 = output_disp2_4 = None
        return output_disp1_1, output_disp1_2, output_disp1_3, output_disp1_4, output_disp2_1, output_disp2_2, output_disp2_3, output_disp2_4
