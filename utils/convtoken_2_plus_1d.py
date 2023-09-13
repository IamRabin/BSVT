
import torch 
import torch.nn as nn
from typing import Callable
from torch.nn.modules.utils import _triple
import math


"""
#class ConvTokenizer(nn.Module):
#    def __init__(
#        self,
#        channels: int = 3, emb_dim: int = 256,
#        conv_kernel: int = 3, conv_stride: int = 2, conv_pad: int = 3,
#        pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 1,
#        activation: Callable = nn.ReLU
#    ):
#        super().__init__()
#        self.conv = nn.Conv2d(
#            in_channels=channels, out_channels=emb_dim,
#            kernel_size=conv_kernel, stride=conv_stride,
#            padding=(conv_pad, conv_pad)
#        )
#        self.act = activation(inplace=True)
#        self.max_pool = nn.MaxPool2d(
#            kernel_size=pool_kernel, stride=pool_stride, 
#            padding=pool_pad
#        )
#            
#    def forward(self, x: torch.Tensor):
#        x = self.conv(x)
#        x = self.act(x)
#        x = self.max_pool(x)
#        return x
#"""



class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x







class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 5, padding: int = 0) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 7),
                stride=(1,3,7),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )



class ConvTokenizer(nn.Module):
    def __init__(
        self,
        channels: int = 1, emb_dim: int = 64, outplanes:int=32,
        conv_stride: int = 5, conv_pad: int = 0,
        pool_kernel: int = 2, pool_stride: int =2 , pool_pad: int = 0,
        activation: Callable = nn.ReLU
    ):
        super().__init__()
        #self.conv = Conv2Plus1D(in_planes=channels, out_planes= emb_dim,midplanes= outplanes )
        self.conv = SpatioTemporalConv(1, 64, [7, 3, 5], stride=[7, 3,5], padding=[0, 0, 0])

        self.act = activation(inplace=True)
        #self.max_pool = nn.MaxPool2d(
         #   kernel_size=pool_kernel, stride=pool_stride, 
          #  padding=pool_pad )
            
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        #x = self.max_pool(x)
        return x

if __name__=="__main__":
    x=torch.rand(2,1,118,19,500)
    c=ConvTokenizer(channels=1)
    a=c(x)
    print(a.shape)

    
