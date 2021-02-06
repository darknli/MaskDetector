from models.interface import get_model
from utils import augmentations
from utils.mask_data import MaskData
from utils.logger import get_logger
from utils.config import conf
from time import time
from os import path, makedirs
from datetime import datetime
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from utils.anchor_modules import Anchor
from utils.process import split_dataset
import torch
from tqdm import tqdm
from torch import optim
from loss.base_loss import NormalLoss
from torch.optim.lr_scheduler import MultiStepLR


def burnin_schedule(i):
    if i < conf.warmup_steps[0]:
        factor = pow(i / conf.warmup_steps[0] * 0.1, 4)
    elif i < conf.warmup_steps[1]:
        factor = 0.1
    else:
        factor = i
    return factor


def get_dataloader(logger):
    train_transform = transforms.Compose([
        augmentations.CropResize(conf.size),
        augmentations.Normalization(conf.bgr_mean, (1, 1, 1))
    ])
    val_transform = transforms.Compose([
        augmentations.CropResize(conf.size),
        augmentations.Normalization(conf.bgr_mean, (1, 1, 1))
    ])

    anchor = Anchor(conf.size, conf.num_classes)

    train_dataset = MaskData(conf.images_root, conf.train_path, anchor, transform=train_transform)

    val_dataset = MaskData(conf.images_root, conf.val_path, anchor, transform=val_transform)
    logger.info("train length:{}, val length:{}".format(len(train_dataset), len(val_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                              num_workers=conf.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                            num_workers=conf.num_workers, pin_memory=True)
    return train_loader, val_loader

def run():
    today = str(datetime.fromtimestamp(time())).replace("-", "")[:8] + "_%d" % (time() % 10000)
    workspace = path.join("checkpoint", today)
    if not path.exists(workspace):
        makedirs(workspace)
    logger = get_logger(workspace)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = get_model("base_yolo", conf.num_classes, conf.is_mobile).to(device)
    train_data, val_data = get_dataloader(logger)

    optimizer = optim.Adam(
        model.parameters(),
        lr=conf.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
    )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    for epoch in range(conf.num_epochs):
        with tqdm(train_data) as pbar:
            criterion = NormalLoss(conf.num_classes)
            model.train()
            for data in pbar:
                image = data["image"]
                image = image.to(device)
                pred_list = model(image)
                loss = criterion(pred_list, [data["branch%d" % i].to(device) for i in range(3)])
                loss.backward()
                optimizer.step()
                scheduler.step()
                show_line = criterion.get_result()
                show_line.update({"lr": scheduler.get_lr()[0]})
                pbar.set_postfix(**show_line)

            logger.info("Epoch=%d train " % epoch +
                        "".join(["{}={}".format(k, v) for k, v in criterion.get_result().items()]) +
                        "lr=%.4f" % scheduler.get_lr()[0])

        with tqdm(val_data) as pbar:
            criterion = NormalLoss(conf.num_classes)
            model.eval()
            with torch.no_grad():
                for data in pbar:
                    image = data["image"]
                    image = image.to(device)
                    pred_list = model(image)
                    criterion(pred_list, [data["branch%d" % i].to(device) for i in range(3)])
                    pbar.set_postfix(**criterion.get_result())

            logger.info("Epoch=%d valuation " % epoch +
                        "".join(["{}={}".format(k, v) for k, v in criterion.get_result().items()]) +
                        "lr=%.4f" % scheduler.get_lr()[0])




if __name__ == '__main__':
    run()