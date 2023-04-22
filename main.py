from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import ssl
from model import SmallViT

ssl._create_default_https_context = ssl._create_unverified_context


pl.seed_everything(42)

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

train_dataset = CIFAR10(root="./data",
                        train=True,
                        download=True,
                        transform=train_transform)
val_dataset = CIFAR10(root="./data",
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

model = SmallViT(embed_dim=256,
                 num_channels=3,
                 patch_size=8,
                 num_patches=17,
                 num_classes=10)

pl_trainer = Trainer(accelerator="gpu",
                     max_epochs=100,
                     enable_progress_bar=False,
                     callbacks=[
                         ModelCheckpoint("chkpt",
                                         monitor=SmallViT.VAL_TOP1_ACC_KEY,
                                         mode="max"),
                         WandbLogger(project="CIFAR10",
                                     name="CIFAR10_VIT"),
                     ])
pl_trainer.fit(model=model,
               train_dataloaders=train_loader,
               val_dataloaders=val_loader)
