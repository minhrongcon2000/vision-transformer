import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

from utils import image_to_patch


class ViT(pl.LightningModule):
    TRAIN_LOSS_KEY = "train_loss"
    TRAIN_TOP1_ACC_KEY = "train_top1_acc"
    TRAIN_TOP5_ACC_KEY = "train_top5_acc"

    VAL_LOSS_KEY = "val_loss"
    VAL_TOP1_ACC_KEY = "val_top1_acc"
    VAL_TOP5_ACC_KEY = "val_top5_acc"

    def __init__(self,
                 embed_dim: int,
                 num_channels: int,
                 patch_size: int,
                 num_patches: int,
                 num_classes: int,
                 dim_feedforward: int,
                 num_heads: int,
                 num_transformer_block: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.num_transformer_block = num_transformer_block

        self.input_layer = nn.Linear(
            self.patch_size * self.patch_size * self.num_channels, self.embed_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                     nhead=self.num_heads,
                                                     dim_feedforward=self.dim_feedforward,
                                                     activation=F.gelu),
            num_layers=self.num_transformer_block,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, self.embed_dim)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes)
        )

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
        """Vision transformer feedforward

        Args:
            x (torch.Tensor): image tensor with shape (B, C, H, W)
        """
        x = image_to_patch(
            x, self.patch_size)  # shape: (B, T, E'). T = #patches, E' = P^2 * C
        B, T, _ = x.shape

        x = self.input_layer(x)  # Shape: (B, T, E)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)  # Shape: (B, 1 + T, E)

        x = x + self.pos_encoding[:, :T + 1]

        x = x.transpose(0, 1)  # Shape (1 + T, B, E)
        x = self.transformer(x)

        pred = x[0]

        return self.mlp_head(pred)

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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def on_train_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch}, " + f"Train loss: {self.train_loss}, " +
              f"Train top 1 acc: {self.train_top1_acc.compute()}, " + f"Train top 5 acc: {self.train_top5_acc.compute()}")


if __name__ == "__main__":
    model = ViT(256, 3, 8, 16 + 1, 10)
    img = torch.rand(16, 3, 32, 32)
    print(model(img).shape)
