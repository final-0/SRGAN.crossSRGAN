import argparse
import os
import numpy as np
import math
import itertools
import sys
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_satellite import *
from datasets_satellite import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

os.makedirs("sate_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training") #200->10
parser.add_argument("--dataset_name", type=str, default="road/test", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.7, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=1024, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=1024, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt.lr, opt.b1, opt.b2)
print(4)
cuda = torch.cuda.is_available()

input_shape = (opt.hr_height, opt.hr_width)
hr_shape = (opt.hr_height/2, opt.hr_width/2)

# Initialize generator and discriminator
generator = Generator()
criterion_MSE = torch.nn.MSELoss()

if cuda:
    generator = generator.cuda()
    criterion_MSE = criterion_MSE.cuda()
generator.load_state_dict(torch.load("saved_models/generator.pth"))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=input_shape),
    shuffle=False,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        chunk_dim = 2
        a_x_split = torch.chunk(imgs_lr, chunk_dim, dim=2)

        chunks_lr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_lr.append(c_)
        
        a_x_split = torch.chunk(imgs_hr, chunk_dim, dim=2)

        chunks_hr = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_hr.append(c_)

        gen_hrs = []
        for j in range(len(chunks_lr)):
            gen_hrs.append(generator(chunks_lr[j]))
        
        print(i)
        
        
        up = torch.cat((gen_hrs[0],gen_hrs[1]),3)
        down = torch.cat((gen_hrs[2],gen_hrs[3]),3)            
        gen_hr = torch.cat((up,down),2)
        PSNR1 = 10 * math.log(255*255/criterion_MSE(imgs_hr, gen_hr),10)
        print(PSNR1)
            
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
        
        img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
        save_image(img_grid, "sate_images/%d.png" % batches_done)