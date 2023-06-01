import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from params import par
import torchvision.models as models
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_b, Swin_V2_B_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights
from .common import *
from backmodel.resnet import ChannelAttention, SpatialAttention
from backmodel.convgru import ConvGRU

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)
def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )
def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = inp
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * inp



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

class Reg(nn.Module):
    def __init__(self, inputnum=8) -> None:
        super().__init__()
        self.inputnum = inputnum
        blocknums = [2,2,3,4,6,7,3]
        outputnums = [32,64,64,128,128,256,256]

        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))

        self.inplanes = 32

        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1) # 40 x 28
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1) # 20 x 14
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1) # 10 x 7
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1) # 5 x 4
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1) # 3 x 2
        fcnum = outputnums[6] * 10
        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,6)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,6)


        self.trans1 = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        # for param in self.trans.parameters():
        #     param.requires_grad = False
        self.rot1 = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        # for param in self.rot.parameters():
        #     param.requires_grad = False
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.shape[0], -1)
        trans = self.trans1(x)
        rot = self.rot1(x)
        out = torch.cat((trans[:,:3],rot[:,:3],trans[:,3:],rot[:,3:]),dim=1)
        return out

        # return torch.cat((trans[:,:3],rot[:,:3],trans[:,3:],rot[:,3:]),dim=1)

class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        # self.model =  raft_small(weights=Raft_Small_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, x1,x2):
        # x1 = x[:,0:3,:]
        # x2 = x[:,3:6,:]
        res = self.model(x1,x2)
        return res

class NewNet(nn.Module):
    def __init__(self, feature_extractor, regressor):
        super().__init__()
        self.feature_extractor = feature_extractor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        # fe_out_planes = self.feature_extractor.fc.in_features
        self.regressor = regressor
    
    def forward(self,x1, x2):

        x = self.feature_extractor(x1,x2)
        
        x = x[-1]
        x = torch.cat((x,x1,x2), dim=1)
        x = self.regressor(x)
       
        return x

if __name__ == '__main__':
    feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in feature_extractor.parameters():
        param.requires_grad = False