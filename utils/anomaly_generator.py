import cv2
import numpy as np
# from torchvision.io import read_image, ImageReadMode
# from torchvision.transforms import Resize
from imgaug import augmenters as iaa

import os
import glob
import random

from .common import Const
from .perlin import generate_perlin_noise_2d


class AnomalyGenerator(object):
    def __init__(self, dtd_dir):
        self.dtd_dir = dtd_dir
        self.texture_paths = sorted(
            glob.glob(os.path.join(dtd_dir, "images/*/*.jpg")))
        self.augmenters = [
            iaa.GammaContrast([0.5, 2.0], per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=[0.8, 1.2], add=[-30, 30]),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation([-50, 50], per_channel=True),
            iaa.Solarize(0.5, threshold=[32, 128]),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]
        self.rotate = iaa.Affine(rotate=(-90, 90))

    def gen_ano(self, resize_shape):
        texture_path = random.choice(self.texture_paths)

        texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
        texture = cv2.resize(texture, dsize=list(reversed(resize_shape)))
        # texture = read_image(texture_path, ImageReadMode.RGB)
        # texture = Resize(resize_shape)(texture).numpy().transpose(1, 2, 0)

        seq = iaa.Sequential(random.sample(self.augmenters, 3))
        texture_aug = seq(image=texture)
        return texture_aug

    def gen_mask_01(self, resize_shape):
        scale_x = 2 ** random.randint(*Const.PERLIN_SCALE_RANGE)
        scale_y = 2 ** random.randint(*Const.PERLIN_SCALE_RANGE)

        perlin_noise = generate_perlin_noise_2d(
            resize_shape, [scale_x, scale_y])
        perlin_noise = self.rotate(image=perlin_noise)
        # TODO: should .astype(np.uint8)?
        perlin_mask = (perlin_noise > Const.PERLIN_THRESHOLD).astype(np.uint8)

        return perlin_mask

    def generate(self, img):
        if random.random() < Const.TRAIN_GOOD_PROP:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = np.expand_dims(mask, axis=2)
            img_ano = img
        else:
            mask = self.gen_mask_01(img.shape[:2])
            mask = np.expand_dims(mask, axis=2)
            ano = self.gen_ano(img.shape[:2])
            beta = random.random() * Const.BETA_MAX
            img_ano = ((1 - mask) * img + (1 - beta) * mask * img +
                       beta * mask * ano).astype(np.uint8)
            mask = mask * 255

        tag = "good" if np.sum(mask) == 0 else "bad"
        # print("genersate", mask.dtype, mask.max(), img_ano.dtype, img_ano.max(), img_ano.mean())
        return img_ano, mask, tag
