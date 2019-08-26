import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.use_relu = use_relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.use_relu:
            x = self.relu(x)
        return x

class UNetDownsampleBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        return x

class UNetUpsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, upsampler='bilinear', upscale_factor=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=upscale_factor, bias=True)
        if upsampler == 'pixelshuffle':
            self.upsample = PixelShuffleConv2d(input_channels, output_channels, kernel_size=kernel_size)
        if upsampler == 'bilinear':
            self.upsample = UpsampleBlock2d(input_channels, output_channels)

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat((x, y), dim=1)
        return x

class UpsampleBlock2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.relu(x)
        return x   
class PixelShuffleConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, upscale_factor=2):
        super().__init__()
        _output_channels = input_channels * 4
        self.conv1 = nn.Conv2d(input_channels, _output_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        # encoder
        self.block1 = UNetBlock(input_channels, 8)
        self.block2 = UNetBlock(8, 8)
        self.down_block1 = UNetDownsampleBlock(8)
        self.block3 = UNetBlock(8, 16)
        self.block4 = UNetBlock(16, 16)
        self.down_block2 = UNetDownsampleBlock(16)
        self.block5 = UNetBlock(16, 32)
        self.block6 = UNetBlock(32, 32)
        self.down_block3 = UNetDownsampleBlock(32)
        self.block7 = UNetBlock(32, 64)
        self.block8 = UNetBlock(64, 64)
        self.down_block4 = UNetDownsampleBlock(64)
        self.block9 = UNetBlock(64, 128)
        self.block10 = UNetBlock(128, 128)
        self.down_block5 = UNetDownsampleBlock(128)
        self.block11 = UNetBlock(128, 256)
        self.block12 = UNetBlock(256, 256)
        self.sigmoid = nn.Sigmoid()

        # decoder, comprised of upsampling blocks
        self.up_block1 = UNetUpsampleBlock(256, 128)
        self.block13 = UNetBlock(256, 128)
        self.block14 = UNetBlock(128, 128)

        self.up_block2 = UNetUpsampleBlock(128, 64)
        self.block15 = UNetBlock(128, 64)
        self.block16 = UNetBlock(64, 64)
        self.up_block3 = UNetUpsampleBlock(64, 32)
        self.block17 = UNetBlock(64, 32)
        self.block18 = UNetBlock(32, 32)
        self.up_block4 = UNetUpsampleBlock(32, 16)
        self.block19 = UNetBlock(32, 16)
        self.block20 = UNetBlock(16, 16)
        self.up_block5 = UNetUpsampleBlock(16, 8)
        self.block21 = UNetBlock(16, 8)
        self.block22 = UNetBlock(8, 3, use_relu=False)

    def forward(self, x):
        x1 = self.block2(self.block1(x))
        x = self.down_block1(x1)
        x2 = self.block4(self.block3(x))
        x = self.down_block2(x2)
        x3 = self.block6(self.block5(x))
        x = self.down_block3(x3)
        x4 = self.block8(self.block7(x))
        x = self.down_block4(x4)
        x5 = self.block10(self.block9(x))
        x = self.down_block5(x5)
        x = self.block12(self.block11(x))

        x = self.up_block1(x, x5)
        x = self.block14(self.block13(x))
        x = self.up_block2(x, x4)
        x = self.block16(self.block15(x))
        x = self.up_block3(x, x3)
        x = self.block18(self.block17(x))
        x = self.up_block4(x, x2)
        x = self.block20(self.block19(x))
        x = self.up_block5(x, x1)
        x = self.block22(self.block21(x))
        x = self.sigmoid(x)
        return x