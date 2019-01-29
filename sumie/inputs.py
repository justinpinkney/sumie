import torch
import numpy as np

import math


class RgbImage(torch.nn.Module):
    """An image stored as a set of RGB pixel values."""

    def __init__(self, size):
        """Creates a simple RGB image.

        Args:
            size (int or tuple): the height and width of the image.

        """
        super(RgbImage, self).__init__()
        if not isinstance(size, tuple):
            size = (size, size)

        #TODO allow other initialisations
        noise_scale = 0.1
        self.pixels = torch.nn.Parameter(noise_scale*torch.randn(1, 3, *size))

    def forward(self):
        return self.pixels

# FFT stuff just like in lucid: https://github.com/tensorflow/lucid
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: math.ceil(w/2) + 1]
    return np.sqrt(fx * fx + fy * fy)


class FftImage(torch.nn.Module):

    def __init__(self, shape, decay_power=1):
      super(FftImage, self).__init__()
      h, w = shape
      freqs = rfft2d_freqs(h, w)
      init_val_size = (ch,) + freqs.shape + (2,)

      # Create a random variable holding the actual 2D fourier coefficients
      init_val = np.random.normal(size=init_val_size, scale=0.01)
      self.pixels = torch.nn.Parameter(torch.Tensor(init_val))

      # Scale the spectrum, see lucid
      scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
      scale *= np.sqrt(w * h)
      scale = torch.from_numpy(scale)
      self.scale = scale[None,:,:,None].float()
      self.register_buffer('scale_const', self.scale)
      self.w = w
      self.h = h

    def forward(self):
      scaled_spectrum_t = self.scale_const*self.pixels
      image_t = torch.irfft(scaled_spectrum_t, 2)
      return image_t[None, :, :self.h, :self.w]/4

