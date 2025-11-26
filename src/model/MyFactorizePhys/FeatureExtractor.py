from torch import nn


class ConvolutionNorm(nn.Module):
    def __init__(self, ch_input, ch_output, stride=(1, 1, 1), padding=(1, 0, 0), **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(ch_input, ch_output, (3, 3, 3), stride=stride, padding=padding, bias=False, **kwargs)
        self.activation = nn.Tanh()
        self.norm = nn.InstanceNorm3d(ch_output)
    
    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class FeatureExtractor(nn.Module):
    def __init__(self, ch_input, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            ConvolutionNorm(ch_input, 8, padding=(1, 1, 1)),
            ConvolutionNorm(8, 12, stride=(1, 2, 2)),
            ConvolutionNorm(12, 12),
            nn.Dropout3d(p=dropout),
            ConvolutionNorm(12, 12),
            ConvolutionNorm(12, 16, stride=(1, 2, 2)),
            ConvolutionNorm(16, 16),
            nn.Dropout3d(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)