import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:14])

    def forward(self, img):
        return self.feature_extractor(img)
"""
class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor2 = nn.Sequential(*list(vgg19_model.features.children())[:14])
    
    def forward(self, img):
        return self.feature_extractor2(img)
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        Rl = []
        Rl.append(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1))
        Rl.append(nn.BatchNorm2d(in_features,0.8))
        Rl.append(nn.LeakyReLU())
        Rl.append(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1))
        Rl.append(nn.BatchNorm2d(in_features))
        #Rl.append(nn.LeakyReLU())
        #Rl.append(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1))
        self.model_R = nn.Sequential(*Rl)
    def forward(self, x):
        return x + self.model_R(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
        res_block = []
        for _ in range(4):
            res_block.append(ResidualBlock(64))
        self.res = nn.Sequential(*res_block)        

        self.conv2 = nn.Sequential(nn.Conv2d(64,64,3,1,1))

        upsampling = []
        for _ in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), nn.Conv2d(64,3,3,1,1))

    def forward(self, x):
        out1 = self.conv1(x)
        out1_re = self.res(out1)
        out2 = self.conv2(out1_re)
        out = torch.add(out1,out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        #change in situation
        patch_h, patch_w = int(in_height/16), int(in_width/16)
        self.output_shape = (1, patch_h, patch_w)
        layers = []
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
        #layers.append(nn.BatchNorm2d(64,0.01))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(512, 1, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
