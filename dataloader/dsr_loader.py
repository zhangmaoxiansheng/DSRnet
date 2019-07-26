import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import preprocess
import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path)

def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, img, lr, hr, training, loader=default_loader, dploader= disparity_loader):

        self.img = img
        self.lr = lr
        self.hr = hr
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        img = self.img[index]
        lr = self.lr[index]
        hr = self.hr[index]
        
        img_ = self.loader(img)
        lr_ = self.dploader(lr)
        hr_ = self.dploader(hr)
        
        if self.training:  
            w, h = img_.size
            th, tw = 256, 512
    
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            img_ = img_.crop((x1, y1, x1 + tw, y1 + th))
            lr_ = np.ascontiguousarray(lr_,dtype=np.float32)
            hr_ = np.ascontiguousarray(hr_,dtype=np.float32)

            lr_ = lr_[y1:y1 + th, x1:x1 + tw]
            hr_ = hr_[y1:y1 + th, x1:x1 + tw]


            processed = preprocess.get_transform(augment=False)  
            img_   = processed(img_)
            

            return img_, lr_, hr_
        else:
            w, h = img_.size
            img_ = img_.crop((w-960, h-544, w, h))
            
            processed = preprocess.get_transform(augment=False)  
            img_       = processed(img_)
            

            return img_, lr_, hr_

    def __len__(self):
        return len(self.img)
