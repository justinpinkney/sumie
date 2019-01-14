import torch
import torchvision
import numpy as np

def max_to_avg_pool(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.MaxPool2d):
            new_child = max_to_avg(child)
            model.__setattr__(name, new_child)
        else:
            max_to_avg_pool(child)

def max_to_avg(max):
    avg = torch.nn.AvgPool2d(kernel_size=max.kernel_size,
                             stride=max.stride,
                             padding=max.padding,
                             )
    return avg

def remove_inplace(model):
    for child in model.modules():
        if hasattr(child, 'inplace'):
            child.inplace = False

class LearnableImage(torch.nn.Module):
    def __init__(self, size):
        super(LearnableImage, self).__init__()
        self.pixels = torch.nn.Parameter(0.1*torch.randn(1,3,size,size))

    def forward(self):
        #return self.pixels
        return torch.sigmoid(self.pixels)
