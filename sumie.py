import torch
import torchvision
import numpy as np

import math

class InputImage(torch.nn.Module):

    def __init__(self, base, *sequence):
        super(InputImage, self).__init__()
        self.base = base
        self.sequence = torch.nn.Sequential(*sequence)

    def forward(self):
        return self.sequence(self.base())

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

class Jitter(torch.nn.Module):
    def __init__(self, amount):
        super(Jitter, self).__init__()
        self.amount = amount

    def forward(self, image):
        shiftx = np.random.randint(-self.amount, self.amount)
        shifty = np.random.randint(-self.amount, self.amount)
        return image.roll((shiftx, shifty), (2, 3))

class Scale(torch.nn.Module):
    def __init__(self, amount):
        super(Scale, self).__init__()
        self.amount = amount

    def forward(self, image):
        scale = np.random.uniform(1/self.amount, self.amount)
        return torch.nn.functional.interpolate(image, scale_factor=scale)

# FFT stuff just like in lucid: https://github.com/tensorflow/lucid
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: math.ceil(w/2) + 1]
    return np.sqrt(fx * fx + fy * fy)


class FftImage(torch.nn.Module):

    def __init__(self, shape, decay_power=1):
      super(FftImage, self).__init__()
      h, w, ch = shape
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
      self.ch = ch
      self.w = w
      self.h = h

    def forward(self):
      scaled_spectrum_t = self.scale_const*self.pixels
      image_t = torch.irfft(scaled_spectrum_t, 2)
      #return torch.sigmoid(image_t[None, :self.ch, :self.h, :self.w]/4)
      return image_t[None, :self.ch, :self.h, :self.w]/4

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
