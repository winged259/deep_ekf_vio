import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from backmodel.core.utils.utils import coords_grid, bilinear_sampler, upflow8
from backmodel.core.FlowFormer.common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from backmodel.core.FlowFormer.encoders import twins_svt_large_context, twins_svt_large
from backmodel.core.position_encoding import PositionEncodingSine, LinearPositionEncoding
from backmodel.core.FlowFormer.LatentCostFormer.twins import PosConv
from backmodel.core.FlowFormer.LatentCostFormer.encoder import MemoryEncoder
from backmodel.core.FlowFormer.LatentCostFormer.decoder import MemoryDecoder
from backmodel.core.FlowFormer.LatentCostFormer.cnn import BasicEncoder

class FlowFormer(nn.Module):
    def __init__(self):
        super(FlowFormer, self).__init__()

        self.memory_encoder = MemoryEncoder()
        self.memory_decoder = MemoryDecoder()
        self.context_encoder = twins_svt_large(pretrained=True)


    def forward(self, image1, image2, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/

        data = {}

        context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        return flow_predictions
    
if __name__ == '__main__':
    a = torch.rand(5,3,120,320)
    b = torch.rand(5,3,120,320)
    model = FlowFormer()
    out = model(a,b)
    for i in out:
        print(i.shape)

