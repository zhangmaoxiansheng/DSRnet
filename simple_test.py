from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataloader import listfile as lt_
from dataloader import dsr_loader as DA
from utils.writepfm import write_pfm
from models import *


parser = argparse.ArgumentParser(description='DSRnet_simple_test')
parser.add_argument('--model', default='basic',
                    help='select model')
parser.add_argument('--img', type=str, default='./sr1.png',
                    help='imagepath')
parser.add_argument('--lr', type=str, default='./lr.pfm',
                    help='lr_depth_path')
parser.add_argument('--output_dir', type=str, default='./hr',
                    help='hr_depth_path')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model = basic()

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

if os.path.isdir(args.img):
    test_imgs = os.listdir(args.img)
    test_imgs.sort()
    test_imgs_path = [os.path.join(args.img,d) for d in test_imgs]
    test_lr = os.listdir(args.lr)
    test_lr.sort()
    test_lr_path = [os.path.join(args.lr,d) for d in test_lr]
else:
    test_imgs_path = [args.img]#should be a list otherwise args.img can be regard as ['.''/'....]
    test_lr_path = [args.lr]
print("there are %d files"%(len(test_imgs_path)))
basenames = [os.path.basename(f) for f in test_imgs_path]
basenames = [os.path.splitext(f)[0] for f in basenames]

Testloader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_imgs_path,test_lr_path,None, False), 
        batch_size= 1, shuffle= False, num_workers= 1, drop_last=False)

def test(img,lr):
    model.eval()
    img = Variable(torch.FloatTensor(img))
    lr  = Variable(torch.FloatTensor(lr))  
    if args.cuda:
        img, lr= img.cuda(), lr.cuda()
    lr = torch.unsqueeze(lr,1)
    
    with torch.no_grad():
        res_output = model(img,lr)
    output = res_output[0].squeeze() + lr.squeeze()
    output = output.cpu()
    output = output.numpy()
    return output
def main():
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    for num, (img, lr) in enumerate(Testloader):
        output = test(img,lr)
        print("finished %d"%(num+1))
        plt.imsave(str(os.path.join(args.output_dir,basenames[num]) + "_sr_vis.png"), output, cmap = 'plasma')
        write_pfm(str(os.path.join(args.output_dir,basenames[num]) + "_sr.pfm"),output)
if __name__ == '__main__':
    main()
