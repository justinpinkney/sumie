import torch
import torchvision
import numpy as np

import math

class Image(torch.nn.Module):
    """Contains a paramterised image and tranformations for optimisation.
    """

    def __init__(self):
        """Creates an Image for optimisation.

        Args:
            size (int or tuple):    desired size of the image.
            parameterisation (str): the desired parameterisation
                This should be one of: 'rgb', 'fft', 'cppn'
            decorrelate (bool):     whether to apply a colour decorrelation transform
            limit (str): desired limit method ('sigmoid', 'clip', 'none')
            transforms (list): List of transforms to be applied to the image

        """

        image = sumie.Image(224, 'fft',
                                decorrelate=True,
                                limit='sigmoid',
                                transforms=transforms)
        pass


class InputImage(torch.nn.Module):

    def __init__(self, base, *sequence):
        super(InputImage, self).__init__()
        self.base = base
        self.sequence = torch.nn.Sequential(*sequence)

    def forward(self):
        return self.sequence(self.base())


class DecorrelateColours(torch.nn.Module):

    def __init__(self):
        super(DecorrelateColours, self).__init__()
        correlation = torch.tensor([[0.26,  0.09,  0.02],
                                    [0.27,  0.00, -0.05],
                                    [0.27, -0.09,  0.03]])
        max_norm = torch.max(torch.norm(correlation, dim=0))
        self.register_buffer('correlation_normalised',
                             correlation/max_norm)

    def forward(self, input):
        reshaped_image = input.view([3, -1])
        output = torch.matmul(self.correlation_normalised, reshaped_image)
        return output.view(input.size())
