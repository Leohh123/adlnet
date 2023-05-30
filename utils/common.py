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
    BETA_MAX = 1.0
    W_SSIM = 1.0
    W_FOCAL = 2.0
    AVG_POOL_FILTER_SIZE = 21


def get_model_info(args):
    dir, file = os.path.split(args.model)
    prefix = os.path.splitext(file)[0]
    name, tag = prefix.split("@")
    return dir, name, tag


def gen_model_name(args, class_name=None):
    return "_".join([
        datetime.now().strftime("%m%d%H%M"),
        class_name or Const.CLASS_NAMES[args.classno],
        str(args.epochs),
        str(args.batch_size),
        f"{args.lr:.1e}"
    ])


def get_class_name(args):
    if hasattr(args, "classno") and args.classno is not None:
        return Const.CLASS_NAMES[args.classno]
    try:
        return next(
            s for s in Const.CLASS_NAMES
            if args.model.find(s) != -1
        )
    except:
        raise Exception("Invalid class name")


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Logger(object):
    DEFAULT_FORMAT = "[%(asctime)s %(src_name)s %(levelname)s] (%(logger_name)s) %(message)s"
    DEFAULT_DATEFMT = "%m-%d %H:%M:%S"

    log_dir = None
    root_logger = None
    model_name = None
    model_tag = None

    @classmethod
    def config(cls, action, args, model_name, model_tag=None):
        cls.model_name = model_name
        cls.model_tag = model_tag
        cls.log_dir = os.path.join(args.log_dir, model_name, action)

        if not os.path.exists(cls.log_dir):
            os.makedirs(cls.log_dir)

        Logger.root_logger = logging.getLogger()
        Logger.root_logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            cls.DEFAULT_FORMAT, cls.DEFAULT_DATEFMT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        Logger.root_logger.addHandler(stream_handler)

        log_path = os.path.join(cls.log_dir, "console.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        Logger.root_logger.addHandler(file_handler)

        Logger.root_logger = logging.LoggerAdapter(Logger.root_logger, {
            "src_name": "root",
            "logger_name": "default"
        })

    def __init__(self, src_name):
        self.loggers = {}
        self.src_name = src_name

        self.default_logger = logging.getLogger(src_name)
        self.default_logger = logging.LoggerAdapter(
            self.default_logger, {
                "src_name": src_name,
                "logger_name": "default"
            }
        )

    def setup_logger(self, name):
        logger = logging.getLogger(name)

        log_path = os.path.join(self.log_dir, f"{name}.csv")
        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)

        logger = logging.LoggerAdapter(logger, {
            "src_name": self.src_name,
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
        self.default_logger.info(", ".join(map(str, messages)))


class Picker(object):
    def __init__(self, rule_list):
        self.rules = {}
        self.scores = {}
        self.epochs = {}

        for args in rule_list:
            self.add_rule(*args)

    def add_rule(self, name, fn=lambda x: x, min_val=0.0):
        assert name not in self.rules, "The rule already exists"
        self.rules[name] = fn
        self.scores[name] = min_val
        self.epochs[name] = -1

    def check(self, name, epoch, *args):
        assert name in self.rules, "The rule does not exist"
        val = self.rules[name](*args)
        if val > self.scores[name]:
            self.scores[name] = val
            self.epochs[name] = epoch
            return True
        return False
