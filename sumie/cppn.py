import numpy as np
import torch

def comp_activation(x, unbiased=True):
    """composite activation from lucid cppn"""
    x = torch.atan(x)
    x1 = x/0.67
    x2 = (x*x - 0.45)/0.396
    return torch.cat((x1, x2), dim=1)

class cppn(torch.nn.Module):
    """Creates a default cppn as an input image."""

    def __init__(self, size):
        super(cppn, self).__init__()
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        xx, yy = np.meshgrid(x, y)
        base_input = torch.Tensor(np.dstack((xx, yy))).permute(2, 1, 0).unsqueeze(0)
        self.register_buffer('input', base_input)
        self.net = CPPN(8, 24)

    def forward(self):
        return self.net(self.input)

class CPPN(torch.nn.Module):
    """Implements a compositional pattern producing network."""

    def __init__(self, num_layers, num_channels):
        super(CPPN, self).__init__()
        self.activation = comp_activation
        self.multiplier = 2

        self.inputLayer = torch.nn.Conv2d(2, num_channels, 1)
        torch.nn.init.kaiming_normal_(self.inputLayer.weight, nonlinearity='linear')

        self.innerLayer = torch.nn.ModuleList()
        for i in range(num_layers):
            conv_layer = torch.nn.Conv2d(self.multiplier*num_channels, num_channels, 1)
            torch.nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='linear')
            self.innerLayer.append(conv_layer)

        self.outputLayer = torch.nn.Conv2d(self.multiplier*num_channels, 3, 1)
        torch.nn.init.zeros_(self.outputLayer.weight)

    def forward(self, x):
        
        x = self.activation(self.inputLayer(x))
        
        for layer in self.innerLayer:
            x = self.activation(layer(x))

        x = self.outputLayer(x)
        return x


