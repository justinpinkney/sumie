import numpy as np
import torch
import math

class PositionJitter(torch.nn.Module):
    """Shifts and image in the x and y direction by a random amount.

    Attributes:
        amount (int): maximum amount of shift in pixels

    """

    def __init__(self, amount):
        """Create a jitter transformation my specifing a maximum translation.

        Args:
            amount (int): maximum shift in x and y directions.

        """

        super(PositionJitter, self).__init__()
        self.amount = amount

    def forward(self, image):
        shiftx = np.random.randint(-self.amount, self.amount)
        shifty = np.random.randint(-self.amount, self.amount)
        return image.roll((shiftx, shifty), (2, 3))

class ScaleJitter(torch.nn.Module):
    """Scales an image by a random factor."""

    def __init__(self, amount):
        """Create a scale transform

        Args:
            amount (int or tuple): max amount of scaling to apply.

        Note:
            If specifying a single int for the `amount` then the scaling
            range is assumed to be from (1/amount, amount).

        """

        super(ScaleJitter, self).__init__()
        if not isinstance(amount, tuple):
            amount = (1/amount, amount)
        self.amount = amount

    def forward(self, image):
        scale = np.random.uniform(*self.amount)
        return torch.nn.functional.interpolate(image, scale_factor=scale, mode="bilinear")

class RandomCrop(torch.nn.Module):
    """Randomly crops an image to a desired size."""

    def __init__(self, size):
        """Creates a cropping transform

        Args:
            size (tuple): desired output size (height, width).

        """

        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, image):
        input_size = image.size()
        y_start = np.random.randint(0, input_size[2] - self.size[0])
        y_end = y_start + self.size[0]
        x_start = np.random.randint(0, input_size[3] - self.size[1])
        x_end = x_start + self.size[1]
        return image[:, :, y_start:y_end, x_start:x_end]

class Normalise(torch.nn.Module):
    """Applies imagenet normalisation to an image Tensor."""

    def __init__(self):
        super(Normalise, self).__init__()
        mean = torch.as_tensor([0.485, 0.456, 0.406])
        std = torch.as_tensor([0.229, 0.224, 0.225])
        self.register_buffer('imnet_mean', mean)
        self.register_buffer('imnet_std', std)

    def forward(self, image):
        return (image - self.imnet_mean[None,:,None,None]) / self.imnet_std[None,:,None,None]

class Interpolate(torch.nn.Module):
    """Interpolates and image by a factor."""

    def __init__(self, factor, prefilter=0):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.factor = factor
        self.prefilter = prefilter

    def forward(self, x):
        
        if self.prefilter:
            pool_size = 11

            pre_filt = torch.nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2)
            for i in range(self.prefilter):
                x = pre_filt(x)
        m = torch.nn.functional.interpolate(x, scale_factor=0.1)
        
        
        return self.interp(x, scale_factor=self.factor, mode='bilinear')

class RotationJitter(torch.nn.Module):
    """Applies random rotation."""

    def __init__(self, amount):
        super(RotationJitter, self).__init__()
        self.amount = amount

    def forward(self, image):
        rotation = 2*self.amount*(np.random.rand(1)-0.5)
        values = np.array([[ math.cos(rotation), math.sin(rotation), 0],
             [-math.sin(rotation), math.cos(rotation), 0]])
        affine = torch.Tensor(values.astype(np.float)).unsqueeze(0).to(image.device)
        grid = torch.nn.functional.affine_grid(affine, image.size())
        return torch.nn.functional.grid_sample(image, grid)

class Tile(torch.nn.Module):
    """Tiles the image 2x2"""
    
    def __init__(self):
        super(Tile, self).__init__()
        
    def forward(self, image):
        out = torch.cat((image, image), dim=2)
        return torch.cat((out, out), dim=3)