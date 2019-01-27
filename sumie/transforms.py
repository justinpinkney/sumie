import numpy as np
import torch

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
        return torch.nn.functional.interpolate(image, scale_factor=scale)

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
