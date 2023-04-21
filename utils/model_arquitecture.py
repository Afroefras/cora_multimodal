from torch.optim import Adam
from lightning.pytorch import LightningModule
from torch.nn.functional import binary_cross_entropy
from torch.nn import Sequential, Linear, Sigmoid, Conv1d

class NoiseModel(LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv = Sequential(
            # Conv1d(input_size, 256),
            # Linear(256, num_classes),
            Linear(num_classes, num_classes),
            Sigmoid()
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.conv(x)
        loss = binary_cross_entropy(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

