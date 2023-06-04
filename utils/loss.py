import torch
import torch.nn.functional as F


def focal_loss(input, target, alpha=0.5, gamma=2.0, epsilon=1e-5):
    at = target * alpha + (1.0 - target) * (1.0 - alpha)
    pt = target * input + (1.0 - target) * (1.0 - input)
    loss = -at * (1.0 - pt) ** gamma * (pt + epsilon).log()
    return loss.mean()


def gaussian(window_size, sigma):
    x = torch.arange(0, window_size)
    gauss = (-(x - window_size // 2) ** 2 / (2.0 * sigma ** 2)).exp()
    return gauss / gauss.sum()


def create_window(window_size):
    win1d = gaussian(window_size, 1.5).unsqueeze(1)
    win2d = (win1d @ win1d.t()).unsqueeze(0)
    return win2d


def ssim_loss(img1, img2, window_size=11):
    padd = window_size // 2
    channel = img1.shape[1]
    win = create_window(window_size).to(img1.device)
    filters = win.expand(channel, 1, -1, -1)

    mu1 = F.conv2d(img1, filters, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, filters, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, filters, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, filters, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, filters, padding=padd, groups=channel) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2

    ssim = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) / \
        ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - ssim.mean()
