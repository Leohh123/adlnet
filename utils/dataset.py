import cv2
import numpy as np
from torch.utils.data import Dataset
# from torchvision.io import read_image, ImageReadMode
# from torchvision.transforms import Resize
from imgaug import augmenters as iaa

import os
import glob
from random import random

from .common import Const, Logger
from .anomaly_generator import AnomalyGenerator


class MVTecTrainDataset(Dataset):
    def __init__(self, class_dir, dtd_dir, classno=None, mask_dir=None, with_mask=False, resize_shape=None, transform=None):
        self.logger = Logger(__file__)
        self.class_dir = class_dir
        self.img_paths = sorted(
            glob.glob(os.path.join(class_dir, "train/good/*.png")))
        self.ano_gen = AnomalyGenerator(dtd_dir)
        self.resize_shape = resize_shape
        self.transform = transform
        # img_outline_path = os.path.join(class_dir, "train/outline.png")
        # if os.path.exists(img_outline_path) and with_mask:
        #     self.img_outline = cv2.imread(
        #         img_outline_path, cv2.IMREAD_GRAYSCALE)
        #     if self.resize_shape:
        #         self.img_outline = cv2.resize(
        #             self.img_outline, dsize=list(reversed(self.resize_shape)))
        #     self.logger.info(f"Outlined with image {img_outline_path}")
        # else:
        #     self.img_outline = None
        #     self.logger.info("No outline")

        self.msk = None
        if with_mask:
            self.classno = classno
            self.class_name = Const.CLASS_NAMES[classno]
            single_mask = os.path.join(mask_dir, f"{self.class_name}.png")
            multi_mask = os.path.join(mask_dir, self.class_name)
            if os.path.isfile(single_mask):
                self.msk = single_mask
                # self.msk = cv2.imread(single_mask, cv2.IMREAD_GRAYSCALE)
                # if self.resize_shape:
                #     self.msk = cv2.resize(self.msk, dsize=list(
                #         reversed(self.resize_shape)))
                self.logger.info(f"With single mask {single_mask}")
            elif os.path.isdir(multi_mask):
                self.msk = glob.glob(os.path.join(multi_mask, "*.png"))
                # if self.resize_shape:
                #     self.msk = list(map(lambda im: cv2.resize(
                #         im, dsize=list(reversed(self.resize_shape))), self.msk))
                self.logger.info(f"With multi mask {multi_mask}/*.png")
            else:
                raise Exception("Invalid mask")
        else:
            self.logger.info(f"Without mask")

        # self.rotate = iaa.Sometimes(
        #     Const.TRAIN_ROTEATE_PROP, iaa.Affine(rotate=(-90, 90)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        # print("getitem", img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
        # img = read_image(img_path, ImageReadMode.RGB)
        cur_msk = None
        if isinstance(self.msk, str):
            cur_msk = cv2.imread(self.msk, cv2.IMREAD_GRAYSCALE)
        elif isinstance(self.msk, list):
            cur_msk = next(m for m in self.msk if m.find(img_name) != -1)
            cur_msk = cv2.imread(cur_msk, cv2.IMREAD_GRAYSCALE)
        if self.resize_shape:
            img = cv2.resize(img, dsize=list(reversed(self.resize_shape)))
            if cur_msk is not None:
                cur_msk = cv2.resize(cur_msk, dsize=list(
                    reversed(self.resize_shape)))
            # img = Resize(self.resize_shape)(img)
        # img = img.numpy().transpose(1, 2, 0)

        # print("img.shape", img.shape)
        # img = self.rotate(image=img)
        # print("img(rotated).shape", img.shape)

        img_ano, mask, tag = self.ano_gen.generate(img, cur_msk)

        rotate_prob = random()
        if rotate_prob < Const.TRAIN_ROTEATE_PROP:
            angle = random() * 180.0 - 90.0
            rotate = iaa.Affine(rotate=angle)
            # img = rotate(img)
            # img_ano = rotate(img_ano)
            # mask = rotate(mask)
            img, img_ano, mask = rotate(images=[img, img_ano, mask])
        # print(img_ano.shape, mask.shape)

        if self.transform:
            img = self.transform(img)
            img_ano = self.transform(img_ano)
            mask = self.transform(mask)

        return {
            "img": img,
            "img_ano": img_ano,
            "mask": mask,
            "tag": tag,
            "label": int(tag != "good"),
            "name": img_name
        }


class MVTecTestDataset(Dataset):
    def __init__(self, class_dir, resize_shape=None, transform=None):
        self.class_dir = class_dir
        self.img_paths = sorted(
            glob.glob(os.path.join(class_dir, "test/*/*.png")))
        self.resize_shape = resize_shape
        self.transform = transform
        # print(f"class_dir = {class_dir}")
        # print(f"self.img_paths = {self.img_paths}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        dir_path, img_filename = os.path.split(img_path)
        tag = os.path.basename(dir_path)
        img_name = os.path.splitext(img_filename)[0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
        # img = read_image(img_path, ImageReadMode.RGB)

        if tag == "good":
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            # mask = torch.zeros(1, *img.shape[1:])
            # print(img.shape)
        else:
            mask_filename = img_filename.split(".")[0] + "_mask.png"
            mask_path = os.path.join(
                self.class_dir, "ground_truth", tag, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # mask = read_image(mask_path, ImageReadMode.GRAY)
        # print(index, img.shape, mask.shape)

        if self.resize_shape:
            img = cv2.resize(img, dsize=list(reversed(self.resize_shape)))
            mask = cv2.resize(mask, dsize=list(reversed(self.resize_shape)))
            # img = Resize(self.resize_shape)(img)
            # mask = Resize(self.resize_shape)(mask)
        # print(index, img.shape, mask.shape)

        mask = np.expand_dims(mask, axis=2)
        # print("mask(expand).shape", mask.shape)

        # img = img.numpy().transpose(1, 2, 0)
        # mask = mask.numpy().transpose(1, 2, 0)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        # print("mask(trans).shape", mask.shape)

        # print(index, type(img), type(mask))
        return {
            "img_ano": img,
            "mask": mask,
            "tag": tag,
            "label": int(tag != "good"),
            "name": img_name
        }
