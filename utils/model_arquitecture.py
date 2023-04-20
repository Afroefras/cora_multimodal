from torch.nn import Module, Linear


class NoiseModel(Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
