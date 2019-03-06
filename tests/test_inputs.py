import sumie
import torch
from pathlib import Path

def test_square_rgb():
    """Get a square rgb image"""
    size = 120
    image = sumie.inputs.RgbImage(size)
    
    output = image()
    
    assert output.size() == (1, 3, size, size)

    
def test_rect_rgb():
    """Give a tuple to specify non-square size"""
    size = (123, 456)
    image = sumie.inputs.RgbImage(size)
    
    output = image()
    
    assert output.size() == (1, 3,) + size

def test_zeros_rgb():
    """init with zeros."""
    image = sumie.inputs.RgbImage((100,100), noise=0)

    assert torch.all(image() == 0)

def test_fft_set():
    """Set the pixels of fft image"""
    tolerance = 1e-10
    size = 120
    image = sumie.inputs.FftImage(size)
    
    # Set to zeros
    image.set_pixels(image()*0)
    
    assert torch.all(image() == 0)
    
    # Use a real image for more representative frequency content
    filename = Path(__file__).with_name("tree.jpg")
    target = sumie.io.load_file(filename, size=(size, size))
    image.set_pixels(target)
    
    assert torch.nn.functional.mse_loss(image(), target).item() < tolerance
    
def test_fft_set_identity():
    """Set the image shouldn't change if we set the original pixels"""
    tolerance = 1e-10
    size = 121
    image = sumie.inputs.FftImage(size)
    original = image().detach()
    
    # Set to zeros
    image.set_pixels(original)
    
    assert torch.nn.functional.mse_loss(image(), original).item() < tolerance