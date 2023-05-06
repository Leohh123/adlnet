import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score, average_precision_score

import os
import argparse

from utils.common import Const, Logger, get_model_info, get_class_name
from utils.dataset import MVTecTestDataset
from model.reconstructive_net import ReconstructiveSubNetwork
from model.discriminative_net import DiscriminativeSubNetwork


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", dest="gpu", metavar="G",
                        type=int, default=0, help="GPU ID")
    parser.add_argument("--class", "-c", dest="classno", metavar="C",
                        type=int, help="training class number")
    parser.add_argument("--model", "-m", dest="model", metavar="M",
                        type=str, required=True, help="path to model file (either .rec or .seg file)")
    parser.add_argument("--mvtec-dir", "-md", dest="mvtec_dir", metavar="MD",
                        type=str, default="./dataset/mvtec", help="MVTec AD dataset directory")
    parser.add_argument("--log-dir", "-ld", dest="log_dir", metavar="LD",
                        type=str, default="./log", help="log directory")
    return parser.parse_args()


def test(
    test_dataset: MVTecTestDataset,
    test_dataloader: DataLoader,
    recon_net: ReconstructiveSubNetwork,
    discr_net: DiscriminativeSubNetwork,
    log_img: bool = False
):
    logger = Logger(__file__)

    recon_net.eval()
    discr_net.eval()

    scores_out, scores_gt = [], []
    masks_out, masks_gt = [], []

    logger.info(f"Start testing: {logger.model_name}@{logger.model_tag}")

    for i, batch in enumerate(test_dataloader):
        img_ano = batch["img_ano"].cuda()
        mask = batch["mask"]
        label = batch["label"]

        img_rec = recon_net(img_ano)
        imgs_ano_rec = torch.cat((img_ano, img_rec), dim=1)

        mask_pred = discr_net(imgs_ano_rec)
        mask_sm = torch.softmax(mask_pred, dim=1)
        mask_prob = mask_sm[:, 1:, ...]
        masks_out.append(mask_prob.cpu().detach().numpy())
        masks_gt.append(mask.cpu().detach().numpy().astype(int))

        nu = F.avg_pool2d(
            mask_prob,
            Const.AVG_POOL_FILTER_SIZE,
            stride=1,
            padding=Const.AVG_POOL_FILTER_SIZE//2
        ).max()
        scores_out.append(nu.cpu().detach().numpy())
        scores_gt.append(label.cpu().detach().numpy())

        if log_img:
            img_name = batch["name"]
            logger.images("img_ano", img_ano, img_name, batch=i)
            logger.images("img_rec", img_rec, img_name, batch=i)
            logger.images("mask", mask, img_name, batch=i)
            logger.images("mask_out", mask_prob, img_name, batch=i)

        if i % 10 == 0:
            logger.info(f"{i * 100 / len(test_dataset):.0f}%")

    scores_out = np.stack(scores_out).flatten()
    scores_gt = np.stack(scores_gt).flatten()
    masks_out = np.stack(masks_out).flatten()
    masks_gt = np.stack(masks_gt).flatten()
    # print(masks_out.shape, masks_gt.shape)
    # print(masks_out, masks_gt)
    # print(masks_gt.max())

    auc_img = roc_auc_score(scores_gt, scores_out)
    auc_px = roc_auc_score(masks_gt, masks_out)

    ap_img = average_precision_score(scores_gt, scores_out)
    ap_px = average_precision_score(masks_gt, masks_out)

    logger.info(f"Test complete: {logger.model_name}@{logger.model_tag}")
    logger.info(
        f"AUC Image: {auc_img}",
        f"AP Image: {ap_img}",
        f"AUC Pixel: {auc_px}",
        f"AP Pixel: {ap_px}"
    )

    return auc_img, ap_img, auc_px, ap_px


if __name__ == "__main__":
    args = get_args()

    class_name = get_class_name(args)
    class_dir = os.path.join(args.mvtec_dir, class_name)
    model_dir, model_name, model_tag = get_model_info(args)

    Logger.config("test", args, model_name, model_tag)

    dataset = MVTecTestDataset(
        class_dir=class_dir,
        resize_shape=[256, 256],
        transform=ToTensor()
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    with torch.cuda.device(args.gpu):
        recon_net = ReconstructiveSubNetwork().cuda()
        recon_net.load_state_dict(torch.load(
            os.path.join(model_dir, f"{model_name}@{model_tag.removesuffix('_tune')}.rec")))

        discr_net = DiscriminativeSubNetwork().cuda()
        discr_net.load_state_dict(torch.load(
            os.path.join(model_dir, f"{model_name}@{model_tag}.seg")))

        test(dataset, dataloader, recon_net, discr_net, True)
