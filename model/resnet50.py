import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision.models import resnet50


class ResNet50(pl.LightningModule):
    TRAIN_LOSS_KEY = "train_loss"
    TRAIN_TOP1_ACC_KEY = "train_top1_acc"
    TRAIN_TOP5_ACC_KEY = "train_top5_acc"

    VAL_LOSS_KEY = "val_loss"
    VAL_TOP1_ACC_KEY = "val_top1_acc"
    VAL_TOP5_ACC_KEY = "val_top5_acc"

    def __init__(self,
                 num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.resnet_model = resnet50()
        self.resnet_model.fc = nn.Linear(2048, self.num_classes)

        self.train_top1_acc = Accuracy(num_classes=self.num_classes,
                                       top_k=1,
                                       task="multiclass")
        self.train_top5_acc = Accuracy(num_classes=self.num_classes,
                                       top_k=5,
                                       task="multiclass")
        self.val_top1_acc = Accuracy(num_classes=self.num_classes,
                                     top_k=1,
                                     task="multiclass")
        self.val_top5_acc = Accuracy(num_classes=self.num_classes,
                                     top_k=5,
                                     task="multiclass")

        self.train_loss = None
        self.val_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet_model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, labels = batch
        preds = self.forward(imgs)

        loss = F.cross_entropy(preds, labels)
        self.train_loss = loss.item()

        self.train_top1_acc(preds, labels)
        self.train_top5_acc(preds, labels)

        self.log(self.TRAIN_LOSS_KEY, loss)
        self.log(self.TRAIN_TOP1_ACC_KEY, self.train_top1_acc)
        self.log(self.TRAIN_TOP5_ACC_KEY, self.train_top5_acc)

        return loss

    def validation_step(self, val_batch, val_idx):
        imgs, labels = val_batch
        preds = self.forward(imgs)

        loss = F.cross_entropy(preds, labels)
        self.val_loss = loss.item()

        self.val_top1_acc(preds, labels)
        self.val_top5_acc(preds, labels)

        self.log(self.VAL_LOSS_KEY, loss)
        self.log(self.VAL_TOP1_ACC_KEY, self.val_top1_acc)
        self.log(self.VAL_TOP5_ACC_KEY, self.val_top5_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=3e-4, momentum=0.99)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def on_train_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch}, " + f"Train loss: {self.train_loss}, " +
              f"Train top 1 acc: {self.train_top1_acc.compute()}, " + f"Train top 5 acc: {self.train_top5_acc.compute()}")
