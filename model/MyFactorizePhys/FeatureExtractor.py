import torch
from torch import nn


"""
For now, this code is copied from the authors' repository.

I wish to be able to begin testing without figuring out whatever is going on here.

I will rewrite this all myself another time.
"""


nf = [8, 12, 16]


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):
    def __init__(self, inCh, dropout_rate=0.1, debug=False):
        super(rPPG_FeatureExtractor, self).__init__()
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),  #B, nf[0], 160, 72, 72
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[1], 160, 35, 35
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 31, 31
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[2], 160, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        voxel_embeddings = self.FeatureExtractor(x)
        return voxel_embeddings