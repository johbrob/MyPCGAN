import torchvision.models as models
import torch.nn as nn
from torch.nn.functional import log_softmax
from torchvision.models.alexnet import AlexNet
import torch

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import librosa
import soundfile as sf
import wave
from torch.nn.functional import relu, max_pool1d, sigmoid, log_softmax

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def double_conv(channels_in, channels_out, kernel_size):
    return nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
        nn.utils.weight_norm(nn.Conv2d(channels_out, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        nn.ReLU()
    )


class UNetFilter(nn.Module):
    def __init__(self, channels_in, channels_out, chs=[2, 4, 8, 16, 32], kernel_size = 3, image_width=64, image_height=64,
                 noise_dim=10, activation='sigmoid', nb_classes=2, embedding_dim=16, use_cond=True):
        super().__init__()
        #chs=[32, 64, 128, 256, 512]
        self.use_cond = use_cond
        self.width  = image_width
        self.height = image_height
        self.activation = activation
        self.embed_condition = nn.Embedding(nb_classes, embedding_dim)
        self.d = 16
        self.chs = chs

        # noise projection layer
        print(image_height, image_width, image_height // self.d, image_width // self.d, chs[-1])
        print(image_width // self.d * image_height // self.d * chs[-1])
        print(image_width // self.d, image_height // self.d, chs[-1])
        self.project_noise = nn.Linear(noise_dim, (image_width // self.d) * (image_height // self.d) * chs[-1])

        # condition projection layer
        self.project_cond = nn.Linear(embedding_dim, (image_width // self.d) * (image_height // self.d))

        self.dconv_down1 = double_conv(channels_in, chs[0], kernel_size)
        self.pool_down1  = nn.MaxPool2d(2, stride=2)

        self.dconv_down2 = double_conv(chs[0], chs[1], kernel_size)
        self.pool_down2  = nn.MaxPool2d(2, stride=2)

        self.dconv_down3 = double_conv(chs[1], chs[2], kernel_size)
        self.pool_down3  = nn.MaxPool2d(2, stride=2)

        self.dconv_down4 = double_conv(chs[2], chs[3], kernel_size)
        self.pool_down4  = nn.MaxPool2d(2, stride=2)

        self.dconv_down5 = double_conv(chs[3], chs[4], kernel_size)

        if self.use_cond:
            self.dconv_up5 = double_conv(chs[4]+chs[4]+1+chs[3], chs[3], kernel_size)
        else:
            self.dconv_up5 = double_conv(chs[4]+chs[4]+chs[3], chs[3], kernel_size)
        self.dconv_up4 = double_conv(chs[3]+chs[2], chs[2], kernel_size)
        self.dconv_up3 = double_conv(chs[2]+chs[1], chs[1], kernel_size)
        self.dconv_up2 = double_conv(chs[1]+chs[0], chs[0], kernel_size)
        self.dconv_up1 = nn.Conv2d(chs[0], channels_out, kernel_size=1)

    def forward(self, x, z, cond):

        bsz, chs, wth, hgt = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # B x C x H x W -> B x 8C x H x W -> B x 8C x H/2 x W/2
        conv1_down = self.dconv_down1(x)
        pool1 = self.pool_down1(conv1_down)

        # B x C x H x W -> B x 2C x H x W -> B x 2C x H/2 x W/2
        conv2_down = self.dconv_down2(pool1)
        pool2 = self.pool_down2(conv2_down)

        # B x C x H x W -> B x 2C x H x W -> B x 2C x H/2 x W/2
        conv3_down = self.dconv_down3(pool2)
        pool3 = self.pool_down3(conv3_down)

        # B x C x H x W -> B x 2C x H x W -> B x 2C x H/2 x W/2
        conv4_down = self.dconv_down4(pool3)
        pool4 = self.pool_down4(conv4_down)

        # B x C x H x W -> B x 2C x H x W
        conv5_down = self.dconv_down5(pool4)

        noise = self.project_noise(z)
        # noise = noise.reshape(bsz, 128 * chs, hgh // 16, wth // 16)
        noise = noise.reshape(bsz, self.chs[-1] * chs, wth // self.d, hgt // self.d)

        if self.use_cond:
            cond_emb = self.embed_condition(cond)
            cond_emb = self.project_cond(cond_emb)
            cond_emb = cond_emb.reshape(bsz, 1, wth // 16, hgt // 16)
            conv5_down = torch.cat((conv5_down, noise, cond_emb), dim=1)
        else:
            # conv5_down: B x 128C x H/16 x W/30
            conv5_down = torch.cat((conv5_down, noise), dim=1)

        conv5_up = F.interpolate(conv5_down, size=conv4_down.size()[-2:], mode='nearest')
        # conv5_up = F.interpolate(conv5_down, scale_factor=2, mode='nearest')

        conv5_up = torch.cat((conv4_down, conv5_up), dim=1)
        conv5_up = self.dconv_up5(conv5_up)

        conv4_up = F.interpolate(conv5_up, size=conv3_down.size()[-2:], mode='nearest')
        # conv4_up = F.interpolate(conv5_up, scale_factor=2, mode='nearest')
        conv4_up = torch.cat((conv3_down, conv4_up), dim=1)
        conv4_up = self.dconv_up4(conv4_up)

        conv3_up = F.interpolate(conv4_up, size=conv2_down.size()[-2:], mode='nearest')
        # conv3_up = F.interpolate(conv4_up, scale_factor=2, mode='nearest')
        conv3_up = torch.cat((conv2_down, conv3_up), dim=1)
        conv3_up = self.dconv_up3(conv3_up)

        conv2_up = F.interpolate(conv3_up, size=conv1_down.size()[-2:], mode='nearest')
        # conv2_up = F.interpolate(conv3_up, scale_factor=2, mode='nearest')
        conv2_up = torch.cat((conv1_down, conv2_up), dim=1)
        conv2_up = self.dconv_up2(conv2_up)

        conv1_up = self.dconv_up1(conv2_up)

        out = torch.tanh(conv1_up)

        return out

class AudioNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #conv3-100, conv3-100, maxpool2, conv3-64, maxpool2, conv3-128, maxpool2, conv3-128, maxpool2, conv3-128, maxpool2, conv3-128, maxpool2, FC-1024, FC-512,

        self.conv1 = nn.Sequential(
                            nn.Conv1d(1, 100, 3, padding = 1),
                            nn.ReLU(inplace = True),
                            nn.MaxPool1d(kernel_size = 2, stride = 2)
                            )
        self.conv2 = nn.Sequential(
                            nn.Conv1d(100, 64, 3, padding = 1),
                            nn.ReLU(inplace = True),
                            nn.MaxPool1d(kernel_size = 2, stride = 2)
                            )

        self.conv3 = nn.Sequential(
                            nn.Conv1d(64, 128, 3, padding = 1),
                            nn.ReLU(inplace = True),
                            nn.MaxPool1d(kernel_size = 2, stride = 2)
                            )

        self.conv4 = nn.Sequential(
                            nn.Conv1d(128, 128, 3, padding = 1),
                            nn.ReLU(inplace = True),
                            nn.MaxPool1d(kernel_size = 2, stride = 2)
                            )

        self.conv5 = nn.Sequential(
                            nn.Conv1d(128, 128, 3, padding = 1),
                            nn.ReLU(inplace = True),
                            nn.MaxPool1d(kernel_size = 2, stride = 2)
                            )

        self.conv6 = nn.Sequential(
                            nn.Conv1d(128, 64, 3, padding = 1),
                            nn.ReLU(inplace = True),
                            nn.MaxPool1d(kernel_size = 2, stride = 2)
                            )

        self.fc1 = nn.Sequential(nn.Linear(8192,1024),nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024,512),nn.Dropout(0.5))
        self.fc3 = nn.Linear(512,num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        conv_out = self.conv6(out)
        out = conv_out.view(-1,8192)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, conv_out



def load_modified_AlexNet(num_classes):
    model = models.AlexNet(num_classes = num_classes)

    # Make single input channel
    model.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    # Change number of output classes to num_classes
    model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num_classes)
        )

    return model

class FID_AlexNet(AlexNet):
    def __init__(self, num_classes):
        super(FID_AlexNet, self).__init__(num_classes)
        # Change to single input channel
        self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        # Change to num_classes output classes
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,num_classes)
            )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        return out

def load_modified_ResNet(num_classes):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
    return model


