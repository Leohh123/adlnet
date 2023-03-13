import torch
import torch.nn as nn


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, base=64):
        super().__init__()
        self.encoder = DiscriminativeEncoder(in_channels, base)
        self.decoder = DiscriminativeDecoder(base, out_channels)

    def forward(self, x):
        bs = self.encoder(x)
        mask_out = self.decoder(*bs)
        return mask_out


class DiscriminativeEncoder(nn.Module):
    def __init__(self, in_channels, base):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(base, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),
            nn.Conv2d(base*4, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(base*4, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True)
        )
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = nn.Sequential(
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True)
        )
        self.mp5 = nn.MaxPool2d(2)
        self.block6 = nn.Sequential(
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True)
        )

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DiscriminativeDecoder(nn.Module):
    def __init__(self, base, out_channels):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base*(8+8), base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*8, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base*(4+8), base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),
            nn.Conv2d(base*4, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*4, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base*(2+4), base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*2, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base*(2+1), base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )

        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True))
        self.db5 = nn.Sequential(
            nn.Conv2d(base*2, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )

        self.outc = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)

    def forward(self, b1, b2, b3, b4, b5, b6):
        up1 = self.up1(b6)
        cat1 = torch.cat((up1, b5), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b4), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b3), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b2), dim=1)
        db4 = self.db4(cat4)

        up5 = self.up5(db4)
        cat5 = torch.cat((up5, b1), dim=1)
        db5 = self.db5(cat5)

        out = self.outc(db5)
        return out
