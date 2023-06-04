import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score, average_precision_score

import os
import argparse

from utils.common import Const, Logger, Picker, get_model_info, init_weights
from utils.dataset import MVTecTrainDataset, MVTecTestDataset
from model.reconstructive_net import ReconstructiveSubNetwork
from model.discriminative_net import DiscriminativeSubNetwork
from utils.loss import focal_loss, ssim_loss

from test import test


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", "-g", dest="gpu", metavar="G",
                        type=int, default=0, help="GPU ID")
    parser.add_argument("--epochs", "-e", dest="epochs", metavar="E",
                        type=int, default=700, help="number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B",
                        type=int, default=8, help="batch size")
    parser.add_argument("--learning-rate", "-l", dest="lr", metavar="LR",
                        type=float, default=1e-4, help="learning rate")
    parser.add_argument("--class", "-c", dest="classno", metavar="C",
                        type=int, default=-1, help="training class number")
    parser.add_argument("--mvtec-dir", "-md", dest="mvtec_dir", metavar="MD",
                        type=str, default="./dataset/mvtec", help="MVTec AD dataset directory")
    parser.add_argument("--dtd-dir", "-dd", dest="dtd_dir", metavar="DD",
                        type=str, default="./dataset/dtd", help="DTD dataset directory")
    parser.add_argument("--checkpoint-dir", "-cd", dest="checkpoint_dir", metavar="CD",
                        type=str, default="./checkpoint", help="checkpoint directory")
    parser.add_argument("--log-dir", "-ld", dest="log_dir", metavar="LD",
                        type=str, default="./log", help="log directory")
    parser.add_argument("--num-workers", "-n", help="number of workers",
                        type=int, default=0, dest="num_workers", metavar="N")
    return parser.parse_args()


def train(
    train_dataset: MVTecTrainDataset,
    train_dataloader: DataLoader,
    test_dataset: MVTecTestDataset,
    test_dataloader: DataLoader,
    recon_net: ReconstructiveSubNetwork,
    discr_net: DiscriminativeSubNetwork,
):
    logger = Logger(__file__)

    recon_net.train()
    discr_net.train()

    optimizer = torch.optim.Adam([
        {"params": recon_net.parameters(), "lr": args.lr},
        {"params": discr_net.parameters(), "lr": args.lr}
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [args.epochs*0.6, args.epochs*0.8], gamma=0.2)
    fn_l2 = torch.nn.MSELoss()

    def save_model(tag):
        torch.save(
            recon_net.state_dict(),
            os.path.join(args.checkpoint_dir, f"{model_name}@{tag}_noperlin.rec")
        )
        torch.save(
            discr_net.state_dict(),
            os.path.join(args.checkpoint_dir, f"{model_name}@{tag}_noperlin.seg")
        )

    rules = [
        ["ws", lambda auc_img, auc_px, ap_px: (auc_img + auc_px) * 5 + ap_px],
        ["sum", lambda auc_img, auc_px, ap_px: auc_img + auc_px + ap_px],
        ["auc-img", lambda auc_img, auc_px, ap_px: auc_img],
        ["auc-px", lambda auc_img, auc_px, ap_px: auc_px],
        ["ap-px", lambda auc_img, auc_px, ap_px: ap_px]
    ]
    pk = Picker(rules)

    logger.info(f"Start training: {logger.model_name}")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}")
        losses = []

        for i, batch in enumerate(train_dataloader):
            img = batch["img"].cuda()
            img_ano = batch["img_ano"].cuda()
            mask = batch["mask"].cuda()

            img_rec = recon_net(img_ano)
            imgs_ano_rec = torch.cat((img_ano, img_rec), dim=1)

            mask_pred = discr_net(imgs_ano_rec)
            mask_sm = torch.softmax(mask_pred, dim=1)
            mask_prob = mask_sm[:, 1:, ...]

            loss_l2 = fn_l2(img_rec, img)
            loss_ssim = ssim_loss(img_rec, img)
            loss_focal = focal_loss(mask_prob, mask, alpha=0.75)

            loss = loss_l2 + Const.W_SSIM * loss_ssim + Const.W_FOCAL * loss_focal

            logger.info(
                f"epoch: {epoch}",
                f"batch: {i}",
                f"loss: {loss.item()}",
                f"l2: {loss_l2.item()}",
                f"ssim: {loss_ssim.item()}",
                f"focal: {loss_focal.item()}"
            )
            losses.append([
                loss.item(), loss_l2.item(),
                loss_ssim.item(), loss_focal.item()
            ])

            if (epoch == 1 and i % 5 == 0) or (epoch % 20 == 0 and i % 20 == 0):
                img_name = batch["name"]
                logger.info("Save images...")
                logger.images("img", img, img_name, epoch, i)
                logger.images("img_ano", img_ano, img_name, epoch, i)
                logger.images("img_rec", img_rec, img_name, epoch, i)
                logger.images("mask", mask, img_name, epoch, i)
                logger.images("mask_prob", mask_prob, img_name, epoch, i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f"epoch {epoch}: avg_loss = {loss_count / cnt}")
        avg, avg_l2, avg_ssim, avg_focal = np.array(
            losses).mean(axis=0).tolist()
        losses.clear()

        logger.scalars("loss", [epoch, avg])
        logger.scalars("loss_l2", [epoch, avg_l2])
        logger.scalars("loss_ssim", [epoch, avg_ssim])
        logger.scalars("loss_focal", [epoch, avg_focal])

        scheduler.step()

        # test
        if epoch % 20 == 0:
            auc_img, ap_img, auc_px, ap_px = test(
                test_dataset, test_dataloader, recon_net, discr_net, False)
            recon_net.train()
            discr_net.train()
            logger.scalars(
                "result", [epoch, auc_img, ap_img, auc_px, ap_px])
            for name, fn in rules:
                if pk.check(name, epoch, auc_img, auc_px, ap_px):
                    save_model(name)
                    logger.info(f"save_model with tag: {name}")
            save_model("last")

    for name, fn in rules:
        logger.scalars("epoch", [name, pk.epochs[name]])
    save_model("last")

    logger.info(f"Training complete: {logger.model_name}")


if __name__ == "__main__":
    args = get_args()

    class_name = Const.CLASS_NAMES[args.classno]
    class_dir = os.path.join(args.mvtec_dir, class_name)
    model_name = get_model_info(args)

    Logger.config("train", args, model_name)

    train_dataset = MVTecTrainDataset(
        class_dir=class_dir,
        dtd_dir=args.dtd_dir,
        resize_shape=[256, 256],
        transform=ToTensor()
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_dataset = MVTecTestDataset(
        class_dir=class_dir,
        resize_shape=[256, 256],
        transform=ToTensor()
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    )

    with torch.cuda.device(args.gpu):
        recon_net = ReconstructiveSubNetwork().cuda()
        recon_net.apply(init_weights)

        discr_net = DiscriminativeSubNetwork().cuda()
        discr_net.apply(init_weights)

        train(
            train_dataset, train_dataloader,
            test_dataset, test_dataloader,
            recon_net, discr_net
        )
