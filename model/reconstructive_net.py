import torch
import torch.nn as nn


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=128):
        super().__init__()
        self.encoder = ReconstructiveEncoder(in_channels, base)
        self.decoder = ReconstructiveDecoder(base, out_channels)

    def forward(self, x):
        latent = self.encoder(x)
        img_recon = self.decoder(latent)
        return img_recon


class ReconstructiveEncoder(nn.Module):
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
        return b5


class ReconstructiveDecoder(nn.Module):
    def __init__(self, base, out_channels):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True)
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*8),
            nn.ReLU(True),
            nn.Conv2d(base*8, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*4, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base*4, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True),
            nn.Conv2d(base*4, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base*2, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base*2, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )

        self.outc = nn.Conv2d(
            base, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.outc(db4)
        return out
