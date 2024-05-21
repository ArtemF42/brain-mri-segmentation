import torch
import torch.nn as nn


class _UNetConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        self.conv1 = _UNetConv(in_channels, 64)
        self.conv2 = _UNetConv(64, 128)
        self.conv3 = _UNetConv(128, 256)
        self.conv4 = _UNetConv(256, 512)
        self.conv5 = _UNetConv(512, 1024)


        self.conv6 = _UNetConv(1024, 512)
        self.conv7 = _UNetConv(512, 256)
        self.conv8 = _UNetConv(256, 128)
        self.conv9 = _UNetConv(128, 64)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.outconv = nn.Conv2d(64, out_channels, 1)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x = self.conv5(self.maxpool(x4))

        x = self.conv6(torch.cat((self.deconv1(x), x4), dim=1))
        x = self.conv7(torch.cat((self.deconv2(x), x3), dim=1))
        x = self.conv8(torch.cat((self.deconv3(x), x2), dim=1))
        x = self.conv9(torch.cat((self.deconv4(x), x1), dim=1))

        return self.outconv(x)
    

if __name__ == '__main__':
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)
