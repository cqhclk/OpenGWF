import torch.nn as nn
import torch.nn.functional as F


class PeriodicPad2d(nn.Module):
    """Circular padding in longitude and zero padding in latitude."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x):
        x = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        return F.pad(x, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0)
