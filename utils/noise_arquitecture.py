from torch.optim import Adam
from torch import relu, flatten
from lightning.pytorch import LightningModule
from torch.nn import Conv1d, MaxPool1d, Linear, BCEWithLogitsLoss


class NoiseClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=2)
        self.pool = MaxPool1d(kernel_size=8)
        self.fc1 = Linear(in_features=1536, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.pool(x)
        x = flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        print(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
