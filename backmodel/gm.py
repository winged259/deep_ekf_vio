import torch
import torch.nn as nn
import torch.nn.functional as F

from backmodel.gmflow.backbone import CNNEncoder
from backmodel.gmflow.transformer import FeatureTransformer, FeatureFlowAttention
from backmodel.gmflow.matching import global_correlation_softmax
from backmodel.gmflow.utils import normalize_img, feature_add_position


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # flow propagation with self-attn

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def forward(self, img0, img1,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=False,
                ):
        
        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]


            attn_splits = attn_splits_list[scale_idx]
   
            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            # correlation and softmax
            prod = global_correlation_softmax(feature0, feature1, pred_bidir_flow)
            return prod


if __name__ == '__main__':
    a = torch.rand(32,3,96,320).cuda()
    b = torch.rand(32,3,96,320).cuda()
    # model = GMFlow(num_scales=1).cuda()
    # out = model(a,b)

    c = torch.rand(32,480,480).unsqueeze(1)
