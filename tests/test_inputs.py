import sumie
import torch

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
