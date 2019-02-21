import torch
import torchvision
import numpy as np
import PIL

import math

import sumie

class Image(torch.nn.Module):
    """Contains a paramterised image and tranformations for optimisation.
    """

    def __init__(self, size=224, param='fft', decorrelate=True, limit='sigmoid', transforms=[], init=None):
        """Creates an Image for optimisation.

        Args:
            size (int or tuple):    desired size of the image.
            parameterisation (str): the desired parameterisation
                This should be one of: 'rgb', 'fft', 'cppn'
            decorrelate (bool):     whether to apply a colour decorrelation transform
            limit (str): desired limit method ('sigmoid', 'clip', 'none')
            transforms (list): List of transforms to be applied to the image

        """

        super(Image, self).__init__()
        if param == 'fft':
            self.base_image = sumie.inputs.FftImage((size, size))
        elif param == 'rgb':
            self.base_image = sumie.inputs.RgbImage((size, size))
        elif param == 'cppn':
            self.base_image = sumie.cppn.cppn(size)
        if decorrelate:
            self.decorrelation = DecorrelateColours()
        else:
            self.decorrelation = None
        if limit == 'sigmoid':
            self.limit = torch.nn.Sigmoid()
        elif limit == 'clamp':
            self.limit = lambda im: torch.clamp(im + 0.5, min=0, max=1)
        else:
            self.limit = None
        self.transforms = torch.nn.Sequential(*transforms)

        if init is not None:
            self.initialise(init)

    def forward(self):
        im = self.get_image()
        return self.transforms(im)

    def get_image(self):
        im = self.base_image()
        if self.decorrelation:
            im = self.decorrelation(im)
        if self.limit:
            im = self.limit(im)
        else:
            im = im + 0.5
        return im

    def initialise(self, init):
        # TODO handle failures
        criterion = torch.nn.MSELoss()
        if isinstance(self.base_image, sumie.cppn.cppn):
            lr = 0.01
        else:
            lr = 0.1
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        steps = 200
        for iteration in range(steps):
            optimiser.zero_grad()
            loss = criterion(self.get_image(), init)
            loss.backward()
            optimiser.step()
            

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

class Optimiser():
    """Optimises an Image to some objective."""
    def __init__(self):
        self.history = []

    def run(self, image, model, objective, iterations=256, lr=0.1, output=None):
        optimiser = torch.optim.Adam(image.parameters(), lr=lr)
        
        for i in range(iterations):
            optimiser.zero_grad()
            model(image())
            loss = -objective.objective
            self.history.append(objective.objective.detach().cpu())
            if output:
                self._save_snapshot(output, image, i)
            loss.backward()
            optimiser.step()
        
        # Run model one more time to get final loss
        model(image())
        self.history.append(objective.objective.detach().cpu())
        if output:
            self._save_snapshot(output, image, i+1)
        
    def _save_snapshot(self, output, image, i):
        # TODO track number of saved outputs instead of using iterations
        filename = output.join(f"{i:06}.jpg")
        output_image = image.get_image().detach().cpu().numpy()
        jpg = PIL.Image.fromarray(np.uint8(255*np.squeeze(output_image.transpose((2, 3, 1, 0)))))
        jpg.save(str(filename))
