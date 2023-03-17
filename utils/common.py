import cv2
import torch
import numpy as np

import os
import logging
from datetime import datetime


class Const(object):
    CLASS_NAMES = [
        "bottle", "capsule", "grid", "leather", "pill",
        "tile", "transistor", "zipper", "cable", "carpet",
        "hazelnut", "metal_nut", "screw", "toothbrush", "wood"
    ]
    PERLIN_SCALE_RANGE = [0, 5]
    PERLIN_THRESHOLD = 0.5
    TRAIN_GOOD_PROP = 0.5
    TRAIN_ROTEATE_PROP = 0.3
    BETA_MAX = 0.8
    LAMBDA = 1.0
    AVG_POOL_FILTER_SIZE = 21


def get_path(args):
    if hasattr(args, "model"):
        dir, file = os.path.split(args.model)
        name = os.path.splitext(file)[0]
        return dir, name

    return "_".join([
        datetime.now().strftime("%m%d%H%M"),
        Const.CLASS_NAMES[args.classno],
        str(args.epochs),
        str(args.batch_size),
        f"{args.lr:.1e}"
    ])


class Logger(object):
    DEFAULT_FORMAT = "[%(asctime)s %(app_name)s %(levelname)s] (%(logger_name)s) %(message)s"
    DEFAULT_DATEFMT = "%m-%d %H:%M:%S"

    log_dir = None

    def __init__(self, app_name):
        self.loggers = {}
        self.app_name = app_name

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            self.DEFAULT_FORMAT, self.DEFAULT_DATEFMT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        log_path = os.path.join(self.log_dir, "console.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        self.root_logger = logging.LoggerAdapter(root_logger, {
            "app_name": app_name,
            "logger_name": "default"
        })

    @classmethod
    def config(cls, action, args, model_name):
        cls.log_dir = os.path.join(args.log_dir, model_name, action)

        if not os.path.exists(cls.log_dir):
            os.makedirs(cls.log_dir)

    def setup_logger(self, name):
        log_path = os.path.join(self.log_dir, f"{name}.csv")
        handler = logging.FileHandler(log_path)

        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger = logging.LoggerAdapter(logger, {
            "app_name": self.app_name,
            "logger_name": name
        })

        self.loggers[name] = logger
        return logger

    def get_logger(self, name):
        if name not in self.loggers:
            self.setup_logger(name)
        return self.loggers[name]

    def scalars(self, name, values):
        logger = self.get_logger(name)
        if isinstance(values, (list, tuple)):
            values = ",".join(map(str, values))
        logger.info(str(values))

    def images(self, name, imgs, img_names, epoch="x", batch="x"):
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().detach().numpy() * 256
            imgs = imgs.clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
            if imgs.shape[-1] == 3:
                imgs = imgs[..., [2, 1, 0]]

        for i in range(imgs.shape[0]):
            img_dir = os.path.join(self.log_dir, name)
            img_path = os.path.join(
                img_dir, f"{epoch}_{batch}_{img_names[i]}.png")
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            cv2.imwrite(img_path, imgs[i])

    def info(self, *messages):
        self.root_logger.info(", ".join(map(str, messages)))
