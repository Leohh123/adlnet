import torch
import torch.nn as nn
import torch.nn.functional as F


class Cv2d(nn.Module):
    """
    2D Convolutional layers

    Arguments:
            in_filt {int} -- number of input filters
            out_filt {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {"relu"})
    """

    def __init__(self, in_filt, out_filt, kernel_size, stride=(1, 1), activation="relu"):
        super().__init__()
        self.act = activation
        self.cv = nn.Conv2d(
            in_channels=in_filt, out_channels=out_filt,
            kernel_size=kernel_size, stride=stride, padding="same"
        )
        self.bn = nn.BatchNorm2d(out_filt)

    def forward(self, x):
        x = self.cv(x)
        x = self.bn(x)

        if self.act == "relu":
            return F.relu(x)
        return x


def get_filt_num(out_channels):
    cnt_3x3 = int(out_channels / 6)
    cnt_5x5 = int(out_channels / 3)
    cnt_7x7 = int(out_channels / 2)
    return cnt_3x3, cnt_5x5, cnt_7x7


class MultiResBlock(nn.Module):
    """
    MultiRes Block

    Arguments:
            in_channels {int} -- Number of channels coming into mutlires block
            base {int} -- Number of filters in a corrsponding UNet stage
            alpha {float} -- alpha hyperparameter (default: 1.67)
    """

    def __init__(self, in_channels, base, alpha=1.67):

        super().__init__()
        self.alpha = alpha
        self.W = base * alpha

        cnt_3x3, cnt_5x5, cnt_7x7 = get_filt_num(self.W)
        num_out_filters = cnt_3x3 + cnt_5x5 + cnt_7x7

        self.shortcut = Cv2d(
            in_channels, num_out_filters, kernel_size=(1, 1), activation=None)

        self.conv_3x3 = Cv2d(
            in_channels, cnt_3x3, kernel_size=(3, 3),  activation="relu")

        self.conv_5x5 = Cv2d(
            cnt_3x3, cnt_5x5, kernel_size=(3, 3),  activation="relu")

        self.conv_7x7 = Cv2d(
            cnt_5x5, cnt_7x7, kernel_size=(3, 3),  activation="relu")

        self.batch_norm1 = nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = nn.BatchNorm2d(num_out_filters)

    def forward(self, x):

        sc = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        x = x + sc
        x = self.batch_norm2(x)
        x = F.relu(x)

        return x


class Respath(nn.Module):
    """
    ResPath

    Arguments:
            num_in_filters {int} -- Number of filters going in the respath
            num_out_filters {int} -- Number of filters going out the respath
            respath_length {int} -- length of ResPath
    """

    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(Cv2d(
                    num_in_filters, num_out_filters, kernel_size=(1, 1), activation=None))
                self.convs.append(Cv2d(
                    num_in_filters, num_out_filters, kernel_size=(3, 3),  activation="relu"))

            else:
                self.shortcuts.append(Cv2d(
                    num_out_filters, num_out_filters, kernel_size=(1, 1), activation=None))
                self.convs.append(Cv2d(
                    num_out_filters, num_out_filters, kernel_size=(3, 3),  activation="relu"))

            self.bns.append(nn.BatchNorm2d(num_out_filters))

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = F.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = F.relu(x)

        return x


class DiscriminativeSubNetwork(nn.Module):
    """
    DiscriminativeSubNetwork

    Arguments:
            input_channels {int} -- number of channels in image
            num_classes {int} -- number of segmentation classes
            alpha {float} -- alpha hyperparameter (default: 1.67)

    Returns:
            [keras model] -- MultiResUNet-like model
    """

    def __init__(self, input_channels=6, num_classes=1, alpha=1.67):
        super().__init__()
        self.alpha = alpha

        # Encoder
        self.blk_en1 = MultiResBlock(input_channels, 32)
        self.filt_en1 = sum(get_filt_num(32 * self.alpha))
        self.pool1 = nn.MaxPool2d(2)
        self.respath1 = Respath(self.filt_en1, 32, respath_length=5)

        self.blk_en2 = MultiResBlock(self.filt_en1, 64)
        self.filt_en2 = sum(get_filt_num(64 * self.alpha))
        self.pool2 = nn.MaxPool2d(2)
        self.respath2 = Respath(self.filt_en2, 64, respath_length=4)

        self.blk_en3 = MultiResBlock(self.filt_en2, 128)
        self.filt_en3 = sum(get_filt_num(128 * self.alpha))
        self.pool3 = nn.MaxPool2d(2)
        self.respath3 = Respath(self.filt_en3, 128, respath_length=3)

        self.blk_en4 = MultiResBlock(self.filt_en3, 256)
        self.filt_en4 = sum(get_filt_num(256 * self.alpha))
        self.pool4 = nn.MaxPool2d(2)
        self.respath4 = Respath(self.filt_en4, 256, respath_length=2)

        self.blk_en5 = MultiResBlock(self.filt_en4, 512)
        self.filt_en5 = sum(get_filt_num(512 * self.alpha))
        self.pool5 = nn.MaxPool2d(2)
        self.respath5 = Respath(self.filt_en5, 512, respath_length=1)

        self.blk_en6 = MultiResBlock(self.filt_en5, 512)
        self.filt_en6 = sum(get_filt_num(512 * self.alpha))

        # Decoder
        self.up5 = nn.ConvTranspose2d(
            self.filt_en6, 512, kernel_size=(2, 2), stride=(2, 2))
        self.filt_cat5 = 512 * 2
        self.blk_de5 = MultiResBlock(self.filt_cat5, 512)
        self.filt_de5 = sum(get_filt_num(512 * self.alpha))

        self.up4 = nn.ConvTranspose2d(
            self.filt_de5, 256, kernel_size=(2, 2), stride=(2, 2))
        self.filt_cat4 = 256 * 2
        self.blk_de4 = MultiResBlock(self.filt_cat4, 256)
        self.filt_de4 = sum(get_filt_num(256 * self.alpha))

        self.up3 = nn.ConvTranspose2d(
            self.filt_de4, 128, kernel_size=(2, 2), stride=(2, 2))
        self.filt_cat3 = 128 * 2
        self.blk_de3 = MultiResBlock(self.filt_cat3, 128)
        self.filt_de3 = sum(get_filt_num(128 * self.alpha))

        self.up2 = nn.ConvTranspose2d(
            self.filt_de3, 64, kernel_size=(2, 2), stride=(2, 2))
        self.filt_cat2 = 64 * 2
        self.blk_de2 = MultiResBlock(self.filt_cat2, 64)
        self.filt_de2 = sum(get_filt_num(64 * self.alpha))

        self.up1 = nn.ConvTranspose2d(
            self.filt_de2, 32, kernel_size=(2, 2), stride=(2, 2))
        self.filt_cat1 = 32 * 2
        self.blk_de1 = MultiResBlock(self.filt_cat1, 32)
        self.filt_de1 = sum(get_filt_num(32 * self.alpha))

        self.outc = Cv2d(
            self.filt_de1, num_classes+1, kernel_size=(1, 1), activation=None)

    def forward(self, x):
        x_en1 = self.blk_en1(x)
        x_pool1 = self.pool1(x_en1)
        x_en1 = self.respath1(x_en1)

        x_en2 = self.blk_en2(x_pool1)
        x_pool2 = self.pool2(x_en2)
        x_en2 = self.respath2(x_en2)

        x_en3 = self.blk_en3(x_pool2)
        x_pool3 = self.pool3(x_en3)
        x_en3 = self.respath3(x_en3)

        x_en4 = self.blk_en4(x_pool3)
        x_pool4 = self.pool4(x_en4)
        x_en4 = self.respath4(x_en4)

        x_en5 = self.blk_en5(x_pool4)
        x_pool5 = self.pool5(x_en5)
        x_en5 = self.respath5(x_en5)

        x_en6 = self.blk_en6(x_pool5)

        cat5 = torch.cat([self.up5(x_en6), x_en5], axis=1)
        x_de5 = self.blk_de5(cat5)

        cat4 = torch.cat([self.up4(x_de5), x_en4], axis=1)
        x_de4 = self.blk_de4(cat4)

        cat3 = torch.cat([self.up3(x_de4), x_en3], axis=1)
        x_de3 = self.blk_de3(cat3)

        cat2 = torch.cat([self.up2(x_de3), x_en2], axis=1)
        x_de2 = self.blk_de2(cat2)

        cat1 = torch.cat([self.up1(x_de2), x_en1], axis=1)
        x_de1 = self.blk_de1(cat1)

        out = self.outc(x_de1)

        return out
