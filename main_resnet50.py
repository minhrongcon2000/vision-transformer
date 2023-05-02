import argparse
import os
import ssl
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from model import ResNet50

ssl._create_default_https_context = ssl._create_unverified_context


pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--dataset", required=True, type=str,
                    choices=["CIFAR10", "CIFAR100"])
parser.add_argument("--chkpt", type=str)
args = vars(parser.parse_args())

os.environ["WANDB_API_KEY"] = args.get("wandb_api_key")

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(
            (32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784]),
    ]
)

train_dataset = eval(args.get("dataset"))(root="./data",
                                          train=True,
                                          download=True,
                                          transform=train_transform)
val_dataset = eval(args.get("dataset"))(root="./data",
                                        train=False,
                                        download=True,
                                        transform=test_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_dataset,
                        batch_size=256,
                        shuffle=False,
                        num_workers=4)

model = ResNet50(
    num_classes=100 if args.get("dataset") == "CIFAR100" else 10
)

if args["chkpt"] is not None:
    mlp_head_keys = ['resnet_model.fc.weight', 'resnet_model.fc.bias']
    model_dict: OrderedDict = torch.load(args["chkpt"])['state_dict']

    for key in mlp_head_keys:
        model_dict.pop(key)

    model.load_state_dict(model_dict, strict=False)

pl_trainer = Trainer(accelerator="gpu",
                     max_epochs=100,
                     enable_progress_bar=False,
                     callbacks=[
                         ModelCheckpoint("chkpt",
                                         monitor=ResNet50.VAL_TOP1_ACC_KEY,
                                         mode="max"),
                     ],
                     logger=WandbLogger(project=args.get("dataset"),
                                        name=f"{args.get('dataset')}_ResNet50"))
pl_trainer.fit(model=model,
               train_dataloaders=train_loader,
               val_dataloaders=val_loader)
