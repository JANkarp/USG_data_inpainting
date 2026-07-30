import torch
import torch.nn as nn
import torch.nn.functional as F

#The U-net implementation
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        if features is None:
            features = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Budowanie Enkodera
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Budowanie Dekodera
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        #Warstwa końcowa
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- ENKODER ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- BOTTLENECK ---
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # --- DEKODER ---
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            concat_x = torch.cat((skip_connection, x), dim=1)

            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)