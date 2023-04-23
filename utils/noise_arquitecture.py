from torch import flatten
from torch.optim import Adam
from collections import OrderedDict
from lightning.pytorch import LightningModule
from torch.nn.functional import binary_cross_entropy
from torch.nn import (
    Sequential,
    ReLU,
    LazyConv2d,
    MaxPool2d,
    LazyLinear,
    Sigmoid,
    BCEWithLogitsLoss,
)


class NoiseClassifier(LightningModule):
    def __init__(self):
        super().__init__()

        self.conv_net = Sequential(
            OrderedDict(
                [
                    ("C1", LazyConv2d(128, kernel_size=(5, 5))),
                    ("Relu1", ReLU()),
                    ("S2", MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C3", LazyConv2d(256, kernel_size=(5, 5))),
                    ("Relu3", ReLU()),
                    ("S4", MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C5", LazyConv2d(500, kernel_size=(5, 5))),
                    ("Relu5", ReLU()),
                ]
            )
        )

        self.fully_connected = Sequential(
            OrderedDict(
                [
                    ("F6", LazyLinear(450)),
                    ("Relu6", ReLU()),
                    ("F6", LazyLinear(100)),
                    ("Relu6", ReLU()),
                    ("F7", LazyLinear(1)),
                    ("Sigmoid", Sigmoid()),
                ]
            )
        )

    def forward(self, input):
        output = self.conv_net(input)
        output = flatten(output, start_dim=1)
        output = self.fully_connected(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        accuracy = binary_cross_entropy(y_hat, y)
        self.log("train_accuracy", accuracy)

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        accuracy = binary_cross_entropy(y_hat, y)
        self.log("val_accuracy", accuracy)

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
