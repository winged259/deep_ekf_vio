import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backmodel.update import BasicUpdateBlock, SmallUpdateBlock
from backmodel.extractor import BasicEncoder, SmallEncoder
from backmodel.corr import CorrBlock, AlternateCorrBlock
from backmodel.utils.utils import bilinear_sampler, coords_grid, upflow8
from params import par


class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()

        if par.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            par.corr_levels = 4
            par.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            par.corr_levels = 4
            par.corr_radius = 4

        if not par.dropout:
            par.dropout = 0


        # feature network, context network, and update block
        if par.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=par.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=par.dropout)
            self.update_block = SmallUpdateBlock(hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=par.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=par.dropout)
            self.update_block = BasicUpdateBlock(hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if par.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=par.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=par.corr_radius)


        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow, delta_reg = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            
            flow_predictions.append(delta_reg)
            
        return flow_predictions
    
if __name__ == '__main__':
   
    import os, sys; 
    sys.path.append(os.path.dirname(os.path.realpath('/mnt/data/teamAI/duy/deep_ekf_vio/backmodel/utils/utils.py')))
    model = RAFT()
    img1 = torch.rand(5,3,96,320)
    img2 = torch.rand(5,3,96,320)
    iter = 12
    out = model(img1,img2, iter)
    for i in out:
        print(i.shape)