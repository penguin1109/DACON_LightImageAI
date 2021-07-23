import random, os, sys, cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.cuda import amp
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0')
class UNet(nn.Module):
      def __init__(self):
    super(UNet, self).__init__()

    def block(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
      layers = []
      layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                           stride = stride, padding = padding, bias = bias)]
      layers += [nn.BatchNorm2d(num_features = out_channels)]
      layers += [nn.ReLU(inplace = True)]

      return nn.Sequential(*layers)

    # Contracting_path
    self.enc1_1 = block(in_channels = 3, out_channels = 64) #(64,281,408)
    self.enc1_2 = block(in_channels = 64, out_channels = 64) #(64,281,408)

    self.pool1 = nn.MaxPool2d(kernel_size = 2) #(64,140,204)

    self.enc2_1 = block(in_channels = 64, out_channels = 128) #(128,140, 204)
    self.enc2_2 = block(in_channels = 128, out_channels = 128) #(128,140, 204)

    self.pool2 = nn.MaxPool2d(kernel_size = 2) #(128, 70,102)

    self.enc3_1 = block(in_channels = 128, out_channels = 256) #(256,70,102)
    self.enc3_2 = block(in_channels = 256, out_channels = 256) #(256, 70,102)

    
    # Expansive_path
    self.dec3_1 = block(in_channels = 256, out_channels = 128) #(128, 70,102)

    self.unpool2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 2,
                                      stride = 2, padding = 0, bias = True)
    #(128,h,w) = (128,140, 204)
    
    self.dec2_2 = block(in_channels = 256, out_channels = 128) #(128,h,w)
    self.dec2_1 = block(in_channels = 128, out_channels = 64) #(64,h,w)

    self.unpool1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, 
                                      stride = 2, kernel_size = 2, padding = 0, bias = True,output_padding = (1, 0)) 
    #(64,281,408)
    
    self.dec1_2 = block(in_channels = 128, out_channels = 64)
    self.dec1_1 = block(in_channels = 64, out_channels = 64)
    
    self.out = nn.Conv2d(in_channels = 64, out_channels = 3, stride = 1, kernel_size = 1, padding = 0, bias = True)
    self.fin = nn.Sigmoid()
    
  def forward(self, x):
    enc1_1 = self.enc1_1(x)
    enc1_2 = self.enc1_2(enc1_1)
    pool1 = self.pool1(enc1_2)

    enc2_1 = self.enc2_1(pool1)
    enc2_2 = self.enc2_2(enc2_1)
    pool2 = self.pool2(enc2_2)

    enc3_1 = self.enc3_1(pool2)
    enc3_2 = self.enc3_2(enc3_1)

    dec3_1 = self.dec3_1(enc3_2)

    unpool2 = self.unpool2(dec3_1)
    cat2 = torch.cat((unpool2, enc2_2), dim = 1)
    dec2_2 = self.dec2_2(cat2)
    dec2_1 = self.dec2_1(dec2_2)

    unpool1 = self.unpool1(dec2_1)
    cat1 = torch.cat((unpool1, enc1_2), dim = 1)
    dec1_2 = self.dec1_2(cat1)
    dec1_1 = self.dec1_1(dec1_2)

    output = self.out(dec1_1)
    output = self.fin(output)

    return output

