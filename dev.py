import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score, average_precision_score

import os
import argparse

from utils.common import Const, Logger, get_path
from utils.dataset import MVTecTrainDataset, MVTecTestDataset
from model.reconstructive_net import ReconstructiveSubNetwork
from model.discriminative_net import DiscriminativeSubNetwork
from utils.loss import focal_loss, ssim_loss


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


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dev(args):
    class_name = Const.CLASS_NAMES[args.classno]
    class_dir = os.path.join(args.mvtec_dir, class_name)
    model_name = get_path(args)

    Logger.config("dev", args, model_name)
    logger = Logger(__file__)

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

    recon_net = ReconstructiveSubNetwork().cuda()
    recon_net.apply(init_weights)

    discr_net = DiscriminativeSubNetwork().cuda()
    discr_net.apply(init_weights)

    def test(epoch):
        recon_net.eval()
        discr_net.eval()

        scores_out, scores_gt = [], []
        masks_out, masks_gt = [], []

        logger.info(f"Start testing ({class_name}): {model_name}")

        for i, batch in enumerate(test_dataloader):
            img_ano = batch["img_ano"].cuda()
            mask = batch["mask"].cuda()
            label = batch["label"].cuda()

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

            # from matplotlib import pyplot as plt
            # fig, ax = plt.subplots(2, 2)
            # ax[0][0].imshow(img_ano[0, ...].cpu().detach().numpy().transpose(1, 2, 0))
            # ax[0][1].imshow(img_rec[0, ...].cpu().detach().numpy().transpose(1, 2, 0))
            # ax[1][0].imshow(mask[0, ...].cpu().detach().numpy().transpose(1, 2, 0))
            # ax[1][1].imshow(mask_prob[0, ...].cpu().detach().numpy().transpose(1, 2, 0))
            # plt.show()

            # img_name = batch["name"]
            # logger.images("img_ano", img_ano, img_name, batch=i)
            # logger.images("img_rec", img_rec, img_name, batch=i)
            # logger.images("mask", mask, img_name, batch=i)
            # logger.images("mask_out", mask_prob, img_name, batch=i)

            # TODO: logger with periodic sampling
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

        logger.info("Test complete")
        logger.scalars("result", [epoch, auc_img, ap_img, auc_px, ap_px])
        logger.info(
            f"AUC Image: {auc_img}",
            f"AP Image: {ap_img}",
            f"AUC Pixel: {auc_px}",
            f"AP Pixel: {ap_px}"
        )

    optimizer = torch.optim.Adam([
        {"params": recon_net.parameters(), "lr": args.lr},
        {"params": discr_net.parameters(), "lr": args.lr}
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [args.epochs*0.8, args.epochs*0.9], gamma=0.2, last_epoch=-1)

    fn_l2 = torch.nn.MSELoss()

    logger.info(f"Start training ({class_name}): {model_name}")

    for epoch in range(args.epochs):
        recon_net.train()
        discr_net.train()

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
            loss_focal = focal_loss(mask_prob, mask)

            # TODO: focal loss balancing hyperparameter
            loss = loss_l2 + Const.LAMBDA * loss_ssim + 2 * loss_focal

            logger.info(
                f"loss: {loss.item()}",
                f"l2: {loss_l2.item()}",
                f"ssim: {loss_ssim.item()}",
                f"focal: {loss_focal.item()}"
            )
            losses.append([
                loss.item(), loss_l2.item(),
                loss_ssim.item(), loss_focal.item()
            ])

            if epoch % 20 == 0 and i % 20 == 0:
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

        torch.save(
            recon_net.state_dict(),
            os.path.join(args.checkpoint_dir, f"{model_name}.rec")
        )
        torch.save(
            discr_net.state_dict(),
            os.path.join(args.checkpoint_dir, f"{model_name}.seg")
        )

        # test
        if (epoch + 1) % 50 == 0:
            test(epoch)

    logger.info("Training complete")


if __name__ == "__main__":
    args = get_args()
    with torch.cuda.device(args.gpu):
        dev(args)
