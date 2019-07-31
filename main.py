from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from tensorboardX import SummaryWriter
from dataloader import listfile as lt_
from dataloader import dsr_loader as DA
from models import *
from models.submodule import scale_pyramid
from models.submodule import SSIM

parser = argparse.ArgumentParser(description='DSRnet')
parser.add_argument('--stage',  default='first',
                    help='first or distill')
parser.add_argument('--maxdisp', type=int ,default=144,
                    help='maxium disparity')
parser.add_argument('--mask_disp', type=float ,default=0.25,
                    help='define the mask range')
parser.add_argument('--model', default='itnet',
                    help='select model')
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./trained_model',
                    help='save model')
parser.add_argument('--log_dir', default='/home/zhu-ty/mnt/svr11/runs',
                    help='logdir')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if not os.path.isdir(args.savemodel):
    os.mkdir(args.savemodel)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_img, all_lr, all_hr, test_img, test_lr, test_hr = lt_.dataloader(args.datapath)

Trainimgloader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_img,all_lr,all_hr, True), 
            batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

Testimgloader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_img,all_lr,all_hr, False), 
            batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'basic_res':
    model = basic(args.maxdisp)
elif args.model == 'itnet':
    model = iterativenet(args.maxdisp)
else:
    print('no model')


if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def train(img,lr, hr):
    model.train()
    img   = Variable(torch.FloatTensor(img))
    lr   = Variable(torch.FloatTensor(lr)) 
    lr = torch.unsqueeze(lr,1)
    hr = Variable(torch.FloatTensor(hr))
        

    if args.cuda:
        img, lr, hr= img.cuda(), lr.cuda(), hr.cuda()
        
    disp_gt = scale_pyramid(hr.unsqueeze(1))
    lr_pyramid = scale_pyramid(lr)
    
    #mask.detach_()
        #----
    
    
    if args.model == 'basic_res':
        lr_pyramid = [torch.squeeze(d,1) for d in lr_pyramid]
        disp_gt = [torch.squeeze(d,1) for d in disp_gt]
        res_gt = [disp_gt[i] - lr_pyramid[i] for i in range(4)]

        mask = [torch.abs(res_gt[i]) > args.mask_disp for i in range(4)]
        mask = [m.float() for m in mask]
        mask = [m.detach_() for m in mask]
        optimizer.zero_grad()

        res_output = model(img,lr)
        res_output = [torch.squeeze(d,1) for d in res_output]
        output = [res_output[i] + lr_pyramid[i] for i in range(4)]
            
        o_loss = loss = F.smooth_l1_loss(lr_pyramid[0]*mask[0], disp_gt[0]*mask[0], size_average=True) + F.smooth_l1_loss(lr_pyramid[1]*mask[1], disp_gt[1]*mask[1], size_average=True) + F.smooth_l1_loss(lr_pyramid[2]*mask[2], disp_gt[2]*mask[2], size_average=True) + F.smooth_l1_loss(lr_pyramid[3]*mask[3], disp_gt[3]*mask[3], size_average=True)
        
        l1_res_loss = F.smooth_l1_loss(res_output[0]*mask[0], res_gt[0]*mask[0], size_average=True) + F.smooth_l1_loss(res_output[1]*mask[1], res_gt[1]*mask[1], size_average=True) + F.smooth_l1_loss(res_output[2]*mask[2], res_gt[2]*mask[2], size_average=True) + F.smooth_l1_loss(res_output[3]*mask[3], res_gt[3]*mask[3], size_average=True)
        
        diff_loss = torch.exp(l1_res_loss - o_loss)
        SSIM_loss = torch.mean(SSIM(output[0]*mask[0], disp_gt[0]*mask[0])) + torch.mean(SSIM(output[1]*mask[1], disp_gt[1]*mask[1])) + torch.mean(SSIM(output[2]*mask[2], disp_gt[2]*mask[2])) + torch.mean(SSIM(output[3]*mask[3], disp_gt[3]*mask[3]))
        
        lr_l1_loss = F.smooth_l1_loss( output[0]*mask[0], lr_pyramid[0]*mask[0],size_average=True) + F.smooth_l1_loss( output[1]*mask[1], lr_pyramid[1]*mask[1],size_average=True) +  F.smooth_l1_loss(output[2]*mask[2], lr_pyramid[2]*mask[2], size_average=True) +  F.smooth_l1_loss( output[3]*mask[3], lr_pyramid[3]*mask[3],size_average=True)
        lr_l1_loss  = torch.exp(-lr_l1_loss)
        #loss withou mask
        l1_full_loss = F.smooth_l1_loss(output[0], disp_gt[0], size_average=True) + F.smooth_l1_loss(output[1], disp_gt[1], size_average=True) + F.smooth_l1_loss(output[2], disp_gt[2], size_average=True) + F.smooth_l1_loss(output[3], disp_gt[3], size_average=True)
        SSIM_full_loss = torch.mean(SSIM(output[0], disp_gt[0])) + torch.mean(SSIM(output[1], disp_gt[1])) + torch.mean(SSIM(output[2], disp_gt[2])) + torch.mean(SSIM(output[3], disp_gt[3]))
        #total_loss = 1.5 * l1_res_loss + 1 * SSIM_loss + 2.5 * diff_loss + 0.7 * lr_l1_loss
        total_loss = 1.5 * l1_res_loss + 1 * SSIM_loss + 4 * diff_loss + 0.8 * lr_l1_loss + 0.8 * (l1_full_loss + SSIM_full_loss)
    if args.model == 'itnet':
        lr_pyramid = [torch.squeeze(d,1) for d in lr_pyramid]
        disp_gt = [torch.squeeze(d,1) for d in disp_gt]
        res_gt = [disp_gt[i] - lr_pyramid[i] for i in range(4)]

        mask = [torch.abs(res_gt[i]) > args.mask_disp for i in range(4)]
        mask = [m.float() for m in mask]
        mask = [m.detach_() for m in mask]
        optimizer.zero_grad()

        res_output = model(img,lr)
        res_output = [torch.squeeze(d,1) for d in res_output]
        output = [res_output[i] + lr_pyramid[i] for i in range(4)]
            
        o_loss = loss = F.smooth_l1_loss(lr_pyramid[0]*mask[0], disp_gt[0]*mask[0], size_average=True) + F.smooth_l1_loss(lr_pyramid[1]*mask[1], disp_gt[1]*mask[1], size_average=True) + F.smooth_l1_loss(lr_pyramid[2]*mask[2], disp_gt[2]*mask[2], size_average=True) + F.smooth_l1_loss(lr_pyramid[3]*mask[3], disp_gt[3]*mask[3], size_average=True)
        
        l1_res_loss = F.smooth_l1_loss(res_output[0]*mask[0], res_gt[0]*mask[0], size_average=True) + F.smooth_l1_loss(res_output[1]*mask[1], res_gt[1]*mask[1], size_average=True) + F.smooth_l1_loss(res_output[2]*mask[2], res_gt[2]*mask[2], size_average=True) + F.smooth_l1_loss(res_output[3]*mask[3], res_gt[3]*mask[3], size_average=True)
        
        diff_loss = torch.exp(l1_res_loss - o_loss)
        SSIM_loss = torch.mean(SSIM(output[0]*mask[0], disp_gt[0]*mask[0])) + torch.mean(SSIM(output[1]*mask[1], disp_gt[1]*mask[1])) + torch.mean(SSIM(output[2]*mask[2], disp_gt[2]*mask[2])) + torch.mean(SSIM(output[3]*mask[3], disp_gt[3]*mask[3]))
        
        lr_l1_loss = F.smooth_l1_loss( output[0]*mask[0], lr_pyramid[0]*mask[0],size_average=True) + F.smooth_l1_loss( output[1]*mask[1], lr_pyramid[1]*mask[1],size_average=True) +  F.smooth_l1_loss(output[2]*mask[2], lr_pyramid[2]*mask[2], size_average=True) +  F.smooth_l1_loss( output[3]*mask[3], lr_pyramid[3]*mask[3],size_average=True)
        lr_l1_loss  = torch.exp(-lr_l1_loss)
        #loss withou mask
        l1_full_loss = F.smooth_l1_loss(output[0], disp_gt[0], size_average=True) + F.smooth_l1_loss(output[1], disp_gt[1], size_average=True) + F.smooth_l1_loss(output[2], disp_gt[2], size_average=True) + F.smooth_l1_loss(output[3], disp_gt[3], size_average=True)
        SSIM_full_loss = torch.mean(SSIM(output[0], disp_gt[0])) + torch.mean(SSIM(output[1], disp_gt[1])) + torch.mean(SSIM(output[2], disp_gt[2])) + torch.mean(SSIM(output[3], disp_gt[3]))
        #total_loss = 1.5 * l1_res_loss + 1 * SSIM_loss + 2.5 * diff_loss + 0.7 * lr_l1_loss
        total_loss = 1.5 * l1_res_loss + 1 * SSIM_loss + 4 * diff_loss + 0.8 * lr_l1_loss + 0.8 * (l1_full_loss + SSIM_full_loss)

    total_loss.backward()
    optimizer.step()

    return total_loss.data, l1_res_loss.data, o_loss.data, output[0]

def test(img,lr,hr):
        model.eval()
        img = Variable(torch.FloatTensor(img))
        lr  = Variable(torch.FloatTensor(lr)) 
        hr = Variable(torch.FloatTensor(hr))   
        if args.cuda:
            img, lr, hr= img.cuda(), lr.cuda(), hr.cuda()
        lr = torch.unsqueeze(lr,1)
        hr = torch.unsqueeze(hr,1)
        #---------
        mask = hr < 192
        #----

        with torch.no_grad():
            res_output = model(img,lr)
        output = res_output[0].squeeze(1) + lr.squeeze(1)
        #output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]

        if len(hr[mask])==0:
            loss = 0
        else:
            loss = torch.mean(torch.abs(output[mask]-hr.squeeze(1)[mask]))  # end-point-error

        return loss

def adjust_learning_rate(optimizer, epoch, init_learning_rate):
    lr = init_learning_rate * (0.1 ** (epoch//40))#every 40 epochs the learning rate * 0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    writer = SummaryWriter(log_dir=str(args.log_dir))
    global_step = 0
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch,args.learning_rate)

        ## training ##
        for batch_idx, (img_, lr_, hr_) in enumerate(Trainimgloader):
            start_time = time.time()

            total_loss, l1_loss, loss0, result = train(img_,lr_, hr_)
            global_step += 1
            print('Iter %d total_training loss = %.3f , time = %.2f' %(batch_idx, total_loss, time.time() - start_time))
            print('Iter %d l1 loss = %.3f' %(batch_idx, l1_loss))
            print('Iter %d origin loss = %.3f' %(batch_idx, loss0))
            print('Iter %d progress = %.3f' %(batch_idx, loss0 - l1_loss))
            if batch_idx % 5 == 0:
                writer.add_scalar('loss/total_loss', total_loss, global_step=global_step)
                writer.add_scalar('loss/diff_loss', loss0 - l1_loss, global_step=global_step)
            if batch_idx % 50 == 0:
                writer.add_image('image/origin', torch.index_select(lr_,0,torch.LongTensor([0]))/192, global_step=global_step)#lr_ is torch.float
                writer.add_image('image/ref', torch.index_select(hr_,0,torch.LongTensor([0]))/192, global_step=global_step)#result is torch.cuda.float
                writer.add_image('image/result', torch.index_select(result,0,torch.cuda.LongTensor([0]))/192, global_step=global_step)#result is torch.cuda.float
        if epoch % 5 == 0:
            savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(Trainimgloader),
            }, savefilename)

        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
    writer.close()
	#------------- TEST ------------------------------------------------------------
	# total_test_loss = 0
	# for batch_idx, (img, lr, hr) in enumerate(Testimgloader):
    #     test_loss = test(img,lr, hr)
    #     print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
    #     total_test_loss += test_loss

	# print('total test loss = %.3f' %(total_test_loss/len(Testimgloader)))
	# #----------------------------------------------------------------------------------
	# #SAVE test information
	# savefilename = args.savemodel+'testinformation.tar'
	# torch.save({
	# 	    'test_loss': total_test_loss/len(Testimgloader),
	# 	}, savefilename)


if __name__ == '__main__':
    main()
    
