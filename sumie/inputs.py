import torch
import torch.nn.functional as F
import numpy as np

import math


class RgbImage(torch.nn.Module):
    """An image stored as a set of RGB pixel values."""

    def __init__(self, size, noise=0.01):
        """Creates a simple RGB image.

        Args:
            size (int or tuple): the height and width of the image.
            noise (float optional): scale of random noise.

        """
        super(RgbImage, self).__init__()
        if not isinstance(size, tuple):
            size = (size, size)

        self.pixels = torch.nn.Parameter(noise*torch.randn(1, 3, *size))

    def forward(self):
        return self.pixels

# FFT stuff just like in lucid: https://github.com/tensorflow/lucid


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: math.ceil(w/2) + 1]
    return np.sqrt(fx * fx + fy * fy)


class FftImage(torch.nn.Module):

    def __init__(self, size, decay_power=1):
        super(FftImage, self).__init__()
        if not isinstance(size, tuple):
            size = (size, size)
        h, w = size
        freqs = rfft2d_freqs(h, w)
        init_val_size = (3,) + freqs.shape + (2,)

        # Create a random variable holding the actual 2D fourier coefficients
        init_val = np.random.normal(size=init_val_size, scale=0.01)
        self.pixels = torch.nn.Parameter(torch.Tensor(init_val))

        # Scale the spectrum, see lucid
        scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
        scale *= np.sqrt(w * h)
        scale = torch.from_numpy(scale)
        self.scale = scale[None, :, :, None].float()
        self.register_buffer('scale_const', self.scale)
        self.w = w
        self.h = h

    def forward(self):
        scaled_spectrum_t = self.scale_const*self.pixels
        image_t = torch.irfft(scaled_spectrum_t, 2)
        return image_t[None, :, :self.h, :self.w]/4

    def set_pixels(self, pixels):
        input_size = pixels.shape
        target_size = self.pixels.shape
        pad = (0, target_size[2]*2 - input_size[3] - 1, 0, 0, 0, 0)
        padded_input = torch.nn.functional.pad(pixels*4, pad)
        new_data = torch.rfft(padded_input, 2).squeeze(0)/self.scale_const
        self.pixels.data = new_data


class PyramidImage(torch.nn.Module):
    """Pyramid of different resolution images."""

    def __init__(self, size, noise=0.01, levels=4):
        super(PyramidImage, self).__init__()
        if not isinstance(size, tuple):
            size = (size, size)

        self.pixels = torch.nn.ParameterList()
        for level in range(levels):
            this_size = (x//(2**level) for x in size)
            level_pixels = noise*torch.randn(1, 3, *this_size)
            self.pixels.append(torch.nn.Parameter(level_pixels))

        self.size = size

    def forward(self):
        output = torch.zeros(1, 3, *self.size)
        for level in self.pixels:
            output += F.interpolate(level,
                                    size=self.size,
                                    mode='bilinear',
                                    align_corners=False)
        return output