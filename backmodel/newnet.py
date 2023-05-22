import numpy as np

import torch
from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F
from params import par
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.nn.init import kaiming_normal_, constant_


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
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def convx(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)  # , inplace=True)
        )
def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


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
    def __init__(self, inputnum=2) -> None:
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
        fcnum = outputnums[6] * 12
        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        fc1_covar_t = linear(fcnum, 128)
        fc2_covar_t = linear(128,32)
        fc3_covar_t = linear(32,6)

        # fc1_covar_r = linear(fcnum, 128)
        # fc2_covar_r = linear(128,32)
        # fc3_covar_r = linear(32,3)

        self.trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        # for param in self.trans.parameters():
        #     param.requires_grad = False
        self.rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        # for param in self.rot.parameters():
        #     param.requires_grad = False
        self.covar_t = nn.Sequential(fc1_covar_t, fc2_covar_t,fc3_covar_t)
        # self.covar_r = nn.Sequential(fc1_covar_r, fc2_covar_r,fc3_covar_r)
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
        trans = self.trans(x)
        rot = self.rot(x)
        covar_t = self.covar_t(x)
        # covar_r = self.covar_r(x)
        # covar = torch.cat((covar_t, covar_r),dim=1)
        # covar = torch.cat((trans[:,3:6], rot[:, 3:6]),dim=1)
        out = torch.cat((trans,rot, covar_t), dim=1)
        return out

class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        
    def forward(self, x):
        x1 = x[:,0:3,:]
        x2 = x[:,3:6,:]
        res = self.model(x1,x2)
        return res
    
class FlowNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(514,128)
        self.deconv2 = deconv(258,64)
        self.deconv1 = deconv(130,32)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(514)
        self.predict_flow3 = predict_flow(258)
        self.predict_flow2 = predict_flow(130)
        self.predict_flow1 = predict_flow(40)
        

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
    def forward(self, features):
        flow6       = self.predict_flow6(features[-1])
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), features[-2])
        out_deconv5 = crop_like(self.deconv5(features[-1]), features[-2])

        concat5 = torch.cat((features[-2],out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), features[-3])
        out_deconv4 = crop_like(self.deconv4(concat5), features[-3])

        concat4 = torch.cat((features[-3],out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), features[-4])
        out_deconv3 = crop_like(self.deconv3(concat4), features[-4])

        concat3 = torch.cat((features[-4],out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), features[-5])
        out_deconv2 = crop_like(self.deconv2(concat3), features[-5])

        concat2 = torch.cat((features[-5],out_deconv2,flow3_up),1)
        flow2       = self.predict_flow2(concat2)
        flow2_up    = crop_like(self.upsampled_flow2_to_1(flow2), features[-6])
        out_deconv1 = crop_like(self.deconv1(concat2), features[-6])

        concat1 = torch.cat((features[-6],out_deconv1,flow2_up),1)
        flow1       = self.predict_flow1(concat1)
        return flow1


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms.functional as TF
    import matplotlib.pyplot as plt
    import flow_vis
    img1 = '/mnt/data/teamAI/duy/data/kitti/2011_10_03/2011_10_03_drive_0027_extract/image_02/data/0000000000.png'
    img2 = '/mnt/data/teamAI/duy/data/kitti/2011_10_03/2011_10_03_drive_0027_extract/image_02/data/0000000001.png'
    i1 = torchvision.io.read_image(img1).float()
    i2 = torchvision.io.read_image(img2).float()
    i = torch.cat((i1, i2), dim=0).unsqueeze(0)
    # print(i.shape)
    raft = RAFT()
    # print(sum([np.prod(p.size()) for p in raft.parameters()]))
    out = raft(i)
    for d,i in enumerate(out):
        i = i.squeeze(0)
        flow_color = flow_vis.flow_to_color(i.permute(1,2,0).detach().numpy(), convert_to_bgr=False)
        plt.imsave(f"{d}.png", flow_color)