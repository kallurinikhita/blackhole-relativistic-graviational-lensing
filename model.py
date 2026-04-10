import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(2, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))

        d1 = self.up(e2)
        d1 = torch.relu(self.dec1(d1))

        return self.out(d1)


class LensingModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = SimpleUNet()
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)