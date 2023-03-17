import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import os
import argparse

from utils.common import Const, Logger, get_path
from utils.dataset import MVTecTrainDataset
from model.reconstructive_net import ReconstructiveSubNetwork
from model.discriminative_net import DiscriminativeSubNetwork
from utils.loss import focal_loss, ssim_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", dest="gpu", metavar="G",
                        type=int, default=0, help="GPU ID")
    parser.add_argument("--epochs", "-e", dest="epochs", metavar="E",
                        type=int, default=700, help="number of epochs")
    parser.add_argument("--recon-epochs", "-re", dest="recon_epochs", metavar="RE",
                        type=int, default=None, help="number of epochs (recon_net)")
    parser.add_argument("--discr-epochs", "-de", dest="discr_epochs", metavar="DE",
                        type=int, default=None, help="number of epochs (discr_net)")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B",
                        type=int, default=8, help="batch size")
    parser.add_argument("--learning-rate", "-l", dest="lr", metavar="LR",
                        type=float, default=1e-4, help="learning rate")
    parser.add_argument("--recon-lr", "-rl", dest="recon_lr", metavar="RLR",
                        type=float, default=None, help="learning rate (recon_net)")
    parser.add_argument("--discr-lr", "-dl", dest="discr_lr", metavar="DLR",
                        type=float, default=None, help="learning rate (discr_net)")
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
    return parser.parse_args()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args):
    class_name = Const.CLASS_NAMES[args.classno]
    class_dir = os.path.join(args.mvtec_dir, class_name)
    model_name = get_path(args)

    Logger.config("train", args, model_name)
    logger = Logger(__file__)

    dataset = MVTecTrainDataset(
        class_dir=class_dir,
        dtd_dir=args.dtd_dir,
        resize_shape=[256, 256],
        transform=ToTensor()
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    def train_recon():
        recon_net = ReconstructiveSubNetwork().cuda()
        recon_net.apply(init_weights)

        recon_epochs = args.recon_epochs or args.epochs
        recon_lr = args.recon_lr or args.lr

        optimizer = torch.optim.Adam([
            {"params": recon_net.parameters(), "lr": recon_lr}
        ])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [recon_epochs*0.8, recon_epochs*0.9], gamma=0.2, last_epoch=-1)

        fn_l2 = torch.nn.MSELoss()

        for epoch in range(recon_epochs):
            logger.info(f"Epoch {epoch} (recon)")
            losses = []

            for i, batch in enumerate(dataloader):
                img = batch["img"].cuda()
                img_ano = batch["img_ano"].cuda()

                img_rec = recon_net(img_ano)

                loss_l2 = fn_l2(img_rec, img)
                loss_ssim = ssim_loss(img_rec, img)

                # TODO: focal loss balancing hyperparameter
                loss_rec = loss_l2 + Const.LAMBDA * loss_ssim

                logger.info(
                    f"loss_rec: {loss_rec.item()}",
                    f"l2: {loss_l2.item()}",
                    f"ssim: {loss_ssim.item()}"
                )
                losses.append([
                    loss_rec.item(), loss_l2.item(), loss_ssim.item()
                ])

                if epoch % 20 == 0 and i % 20 == 0:
                    img_name = batch["name"]
                    logger.info("Save images...")
                    logger.images("recon_img", img, img_name, epoch, i)
                    logger.images("recon_img_ano", img_ano, img_name, epoch, i)
                    logger.images("recon_img_rec", img_rec, img_name, epoch, i)

                optimizer.zero_grad()
                loss_rec.backward()
                optimizer.step()

            # print(f"epoch {epoch}: avg_loss = {loss_count / cnt}")
            avg_rec, avg_l2, avg_ssim = np.array(losses).mean(axis=0).tolist()
            losses.clear()

            logger.scalars("loss_rec", [epoch, avg_rec])
            logger.scalars("loss_l2", [epoch, avg_l2])
            logger.scalars("loss_ssim", [epoch, avg_ssim])

            scheduler.step()

            torch.save(
                recon_net.state_dict(),
                os.path.join(args.checkpoint_dir, f"{model_name}.rec")
            )

    def train_discr():
        recon_net = ReconstructiveSubNetwork().cuda()
        recon_net.load_state_dict(torch.load(
            os.path.join(args.checkpoint_dir, f"{model_name}.rec")))
        recon_net.eval()

        discr_net = DiscriminativeSubNetwork().cuda()
        discr_net.apply(init_weights)

        discr_epochs = args.discr_epochs or args.epochs
        discr_lr = args.discr_lr or args.lr

        optimizer = torch.optim.Adam([
            {"params": discr_net.parameters(), "lr": discr_lr}
        ])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [discr_epochs*0.8, discr_epochs*0.9], gamma=0.2, last_epoch=-1)

        for epoch in range(discr_epochs):
            logger.info(f"Epoch {epoch} (discr)")
            losses = []

            for i, batch in enumerate(dataloader):
                img = batch["img"].cuda()
                img_ano = batch["img_ano"].cuda()
                mask = batch["mask"].cuda()

                img_rec = recon_net(img_ano)
                imgs_ano_rec = torch.cat((img_ano, img_rec), dim=1)

                mask_pred = discr_net(imgs_ano_rec)
                mask_sm = torch.softmax(mask_pred, dim=1)
                mask_prob = mask_sm[:, 1:, ...]

                # TODO: focal loss balancing hyperparameter
                loss_seg = focal_loss(mask_prob, mask)

                logger.info(f"loss_seg: {loss_seg.item()}")
                losses.append([loss_seg.item()])

                if epoch % 20 == 0 and i % 20 == 0:
                    img_name = batch["name"]
                    logger.info("Save images...")
                    logger.images("discr_img", img, img_name, epoch, i)
                    logger.images("discr_img_ano", img_ano, img_name, epoch, i)
                    logger.images("discr_img_rec", img_rec, img_name, epoch, i)
                    logger.images("discr_mask", mask, img_name, epoch, i)
                    logger.images("discr_mask_prob", mask_prob,
                                  img_name, epoch, i)

                optimizer.zero_grad()
                loss_seg.backward()
                optimizer.step()

            # print(f"epoch {epoch}: avg_loss = {loss_count / cnt}")
            avg_seg, = np.array(losses).mean(axis=0).tolist()
            losses.clear()

            logger.scalars("loss_seg", [epoch, avg_seg])

            scheduler.step()

            torch.save(
                discr_net.state_dict(),
                os.path.join(args.checkpoint_dir, f"{model_name}.seg")
            )

    logger.info(f"Start training ({class_name}): {model_name}")
    train_recon()
    train_discr()
    logger.info("Training complete")


if __name__ == "__main__":
    args = get_args()
    with torch.cuda.device(args.gpu):
        train(args)
