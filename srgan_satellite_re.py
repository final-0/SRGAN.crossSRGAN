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

os.makedirs("train_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training") #200->10
parser.add_argument("--dataset_name", type=str, default="road/train", help="name of the dataset")
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
print(6)

cuda = torch.cuda.is_available()

input_shape = (opt.hr_height, opt.hr_width)
hr_shape = (opt.hr_height/2, opt.hr_width/2)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_G = torch.nn.BCEWithLogitsLoss()
criterion_MSE = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()


if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_G = criterion_G.cuda()
    criterion_MSE = criterion_MSE.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_BCE = criterion_BCE.cuda()


summary(discriminator,(3,512,512))
summary(generator,(3,128,128))
summary(feature_extractor,(3,512,512))

# Optimizers
optimizer_G1 = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, opt.b2))
optimizer_G2 = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

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


        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        # ------------------
        #  Train Generators
        # ------------------
        
        if batches_done < 50:
            for k in range(len(chunks_lr)):
                optimizer_G1.zero_grad()
                loss_pixel = criterion_MSE(generator(chunks_lr[k]), chunks_hr[k])
                loss_pixel.backward()
                optimizer_G1.step()
            continue
        
        # Generate a high resolution image from low resolution input
        gen_hrs = []
        for j in range(len(chunks_lr)):
            """
            optimizer_G1.zero_grad()
            gen_hrs.append(generator(chunks_lr[j]))
            loss_GAN = criterion_G(discriminator(generator(chunks_lr[j])), valid)
            gen_features = feature_extractor(generator(chunks_lr[j]))
            real_features = feature_extractor(chunks_hr[j])
            loss_content = criterion_MSE(gen_features, real_features.detach())
            loss_pixel = criterion_MSE(generator(chunks_lr[j]), chunks_hr[j])
            # Total loss
            loss_G = 1e-3 * loss_GAN + loss_content + 5e-3 * loss_pixel
            loss_G.backward()
            optimizer_G1.step()
            """

        # ---------------------
        #  Train Discriminator
        # ---------------------
            optimizer_G1.zero_grad()
            gen_hrs.append(generator(chunks_lr[j]))
            loss_GAN = criterion_G(discriminator(generator(chunks_lr[j])), valid) + criterion_G(discriminator(chunks_hr[j]), fake)
            loss_x = criterion_G(discriminator(generator(chunks_lr[j])), fake) + criterion_G(discriminator(chunks_hr[j]), valid)
            gen_features = feature_extractor(generator(chunks_lr[j]))
            real_features = feature_extractor(chunks_hr[j])
            loss_content = criterion_MSE(gen_features, real_features.detach())
            loss_pixel1 = criterion_MSE(generator(chunks_lr[j]), chunks_hr[j])
            # Total loss
            loss_G = loss_GAN + loss_content + loss_pixel1 / 5 - loss_x
            loss_G.backward()
            optimizer_G1.step()

            optimizer_D.zero_grad()
            loss_real = criterion_MSE(discriminator(chunks_hr[j]), valid)
            loss_fake = criterion_MSE(discriminator(generator(chunks_lr[j]).detach()), fake)
            loss_y = criterion_G(discriminator(generator(chunks_lr[j])), valid) + criterion_G(discriminator(chunks_hr[j]), fake)
            gen_features = feature_extractor(generator(chunks_lr[j]))
            real_features = feature_extractor(chunks_hr[j])
            loss_content1 = criterion_MSE(gen_features, real_features.detach())
            loss_pixel2 = criterion_MSE(generator(chunks_lr[j]), chunks_hr[j])
            # Total loss
            loss_D = (loss_real + loss_fake) + 10 - loss_y - loss_content1 - loss_pixel2 / 5
            loss_D.backward()
            optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------)
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item()
            )
        )
        
        if batches_done % opt.sample_interval == 0:
            up = torch.cat((gen_hrs[0],gen_hrs[1]),3)
            down = torch.cat((gen_hrs[2],gen_hrs[3]),3)
            gen_hr = torch.cat((up,down),2)
            PSNR1 = 10 * math.log(255*255/criterion_MSE(imgs_hr, gen_hr),10)
            print(PSNR1)
            #gen_hr = gen_hr.squeeze(0)
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            
            img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr),-1)
            save_image(img_grid, "train_images/%d.png" % batches_done)

torch.save(generator.state_dict(), "saved_models/generator.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")
