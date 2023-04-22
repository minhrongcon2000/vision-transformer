import argparse
import os
import ssl

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from model import ViT

ssl._create_default_https_context = ssl._create_unverified_context


pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_api_key", type=str, required=True)
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

train_dataset = CIFAR100(root="./data",
                         train=True,
                         download=True,
                         transform=train_transform)
val_dataset = CIFAR100(root="./data",
                       train=False,
                       download=True,
                       transform=test_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_dataset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=4)

model = ViT(embed_dim=1024,
            num_channels=3,
            patch_size=16,
            num_patches=4,
            num_classes=100,
            num_heads=16,
            dim_feedforward=4096,
            num_transformer_block=24)

pl_trainer = Trainer(accelerator="gpu",
                     max_epochs=100,
                     enable_progress_bar=False,
                     callbacks=[
                         ModelCheckpoint("chkpt",
                                         monitor=ViT.VAL_TOP1_ACC_KEY,
                                         mode="max"),
                     ],
                     logger=WandbLogger(project="CIFAR100",
                                        name="CIFAR100_VIT"))
pl_trainer.fit(model=model,
               train_dataloaders=train_loader,
               val_dataloaders=val_loader)
