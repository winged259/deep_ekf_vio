from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from params import par
from inverse_warp import inverse_warp
# from backmodel.depth import WildDepth

def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s

def scale_loss(P, G, gamma = 0.9):
    batch = P.shape[0]
    seq = P.shape[1]
    n = batch*seq
    P = P.view(n, -1)
    G = G.view(n, -1)
    # loss = Batch x Loss
    scale = fit_scale(P, G)
    loss = []
    for i in range(n):
        weight = gamma ** (n-i-1)
        err = torch.abs(P[i]-G[i])
        err = torch.mean(err)
        loss.append(weight * err)
    l1 = torch.stack(loss)
    l1_scaled = l1/scale
    l1 = torch.mean(l1)
    return l1

def photometric_reconstruction_loss(tgt_img, ref_img, intrinsics,
                                    depth, pose,
                                    rotation_mode='euler', padding_mode='zeros'):
    batch = tgt_img.shape[0]
    seq = tgt_img.shape[1]
    n = batch*seq
    ch = tgt_img.shape[2]
    h = tgt_img.shape[3]
    w = tgt_img.shape[4]
    tgt_img = tgt_img.view(n, ch, h,w)
    ref_img = ref_img.view(n, ch, h, w)
    pose = pose.view(n,6)
    def one_scale(depth):
        reconstruction_loss = 0
        b, _, h, w = depth.size()

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2], intrinsics[:, 2:]), dim=1)

        current_pose = pose

        ref_img_warped, valid_points = inverse_warp(ref_img_scaled, depth[:,0], current_pose,
                                                    intrinsics_scaled,
                                                    rotation_mode, padding_mode)
        diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

        reconstruction_loss += diff.abs().mean()
        assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss, diff[0]

    total_loss = 0

    loss, diff = one_scale(depth)
    total_loss += loss
    return total_loss, diff

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

if __name__ == '__main__':
    a = torch.rand(2,5,3,416,128)
    b = torch.rand(2,5,3,416,128)
    intrinsic = torch.rand(1,3,3)
    dep = torch.rand(10,1,416,128)
    pose = torch.rand(2,5,6)
    loss, diff = photometric_reconstruction_loss(a,b, intrinsics=intrinsic,depth=dep,pose=pose)
    loss1 = smooth_loss(dep)
    print(type(loss1))
    print(type(loss))