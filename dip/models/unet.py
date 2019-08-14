import torch
import torch.nn as nn

class UNetBasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self,relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x

class UNetDownsampleBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        return x

class UNetUpsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, upsampler='pixelshuffle', upscale_factor=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=upscale_factor, padding=1, bias=False)
        if upsampler == 'pixelshuffle':
            self.upsample = PixelShuffleConv2d(input_channels, output_channels, kernel_size=kernel_size)

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat((x, y), dim=1)
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x

class PixelShuffleConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, upscale_factor=2):
        super().__init__()
        _output_channels = input_channels * 4
        self.conv1 = nn.Conv2d(input_channels, _output_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShufle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=3, upsampler='pixelshuffle'):
        super().__init__()
        # encoder
        self.block1 = UNetBasicBlock(input_channels, 64)
        self.block2 = UNetBasicBlock(64, 64)
        self.down_block1 = UNetDownsampleBlock(64)
        self.block3 = UNetBasicBlock(64, 128)
        self.block4 = UNetBasicBlock(128, 128)
        self.down_block2 = UNetDownsampleBlock(128)
        self.block5 = UNetBasicBlock(128, 256)
        self.block6 = UNetBasicBlock(256, 256)
        self.down_block3 = UNetDownsampleBlock(256)
        self.block7 = UNetBasicBlock(256, 512)
        self.block8 = UNetBasicBlock(512, 512)
        self.down_block4 = UNetDownsampleBlock(512)
        self.block9 = UNetBasicBlock(512, 1024)
        self.block10 = UNetBasicBlock(1024, 1024)

        # decoder, comprised of upsampling blocks
        self.up_block1 = UNetUpsampleBlock(1024, 512)
        self.block11 = UNetBasicBlock(512, 512)
        self.block12 = UNetBasicBlock(512, 512)
        self.up_block2 = UNetUpsampleBlock(512, 256)
        self.block13 = UNetBasicBlock(256, 256)
        self.block14 = UNetBasicBlock(256, 256)
        self.up_block3 = UNetUpsampleBlock(256, 128)
        self.block15 = UNetBasicBlock(128, 128)
        self.block16 = UNetBasicBlock(128, 128)
        self.up_block4 = UNetUpsampleBlock(128, 64)
        self.block17 = UNetBasicBlock(64, 64)
        self.block18 = UNetBasicBlock(64, 3)

    def forward(self, x):
        x1 = self.block2(self.block1(x))
        x = self.down_block1(x1)
        x2 = self.block4(self.block3(x))
        x = self.down_block2(x2)
        x3 = self.block6(self.block5(x))
        x = self.down_block3(x3)
        x4 = self.block8(self.block7(x))
        x = self.down_block(x4)
        x = self.block10(self.block9(x))

        x = self.up_block1(x, x4)
        x = self.block12(self.block11(x))
        x = self.up_block2(x, x3)
        x = self.block14(self.block13(x))
        x = self.up_block3(x, x2)
        x = self.block16(self.block15(x))
        x = self.up_block4(x, x1)
        x = self.block18(self.block17(x))
        return x