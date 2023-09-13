
import torch 
import torch.nn as nn
from typing import Callable



class ConvTokenizer(nn.Module):
    def __init__(
        self,
        channels: int = 1, emb_dim: int = 64,
        conv_kernel: int = 3, conv_stride: int = 3, conv_pad: int = 0,
        pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 0,
        activation: Callable = nn.ReLU ):

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels, out_channels=emb_dim,
            kernel_size=conv_kernel, stride=conv_stride,
            padding=(conv_pad, conv_pad)
        )
        self.act = activation(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, 
            padding=pool_pad
        )
            
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.max_pool(x)
        return x






#class Conv2Plus1D(nn.Sequential):
#    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 2, padding: int = 1) -> None:
#        super().__init__(
#            nn.Conv3d(
#                in_planes,
#                midplanes,
#                kernel_size=(1, 7, 7),
#                stride=(1, stride, stride),
#                padding=(0, padding, padding),
#                bias=False,
#            ),
#            nn.BatchNorm3d(midplanes),
#            nn.ReLU(inplace=True),
#            nn.Conv3d(
#                midplanes, out_planes, kernel_size=(7, 1,1 ), stride=(stride, 1,1 ), padding=(padding, 0, 0), bias=False
#            ),
#        )
#
#
#
#
#class ConvTokenizer(nn.Module):
#    def __init__(
#        self,
#        channels: int = 1, emb_dim: int = 64,outplanes:int=32,
#        conv_stride: int = 1, conv_pad: int = 1,
#        pool_kernel: int = 7, pool_stride: int = 2, pool_pad: int = 1,
#        activation: Callable = nn.ReLU
#    ):
#        super().__init__()
#        self.conv = Conv2Plus1D(
#            in_planes=channels, out_planes= emb_dim,midplanes= outplanes, 
#            stride=conv_stride,padding=conv_pad
#        )
#        self.act = activation(inplace=True)
#        self.max_pool = nn.MaxPool3d(
#            kernel_size=pool_kernel, stride=pool_stride, 
#            padding=pool_pad
#        )
#            
#    def forward(self, x: torch.Tensor):
#        x = self.conv(x)
#        x = self.act(x)
#        x = self.max_pool(x)
#        return x

if __name__=="__main__":
    x=torch.rand(1,118,19,500)
    c=ConvTokenizer(channels=118)
    a=c(x)
    print(a.shape)

    
