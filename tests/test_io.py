import sumie
import torch

from pathlib import Path
import os

def test_im_file():
    """Load an image from file."""
    filename = Path(__file__).with_name("tree.jpg")
    image = sumie.io.load_file(filename)

    assert image.size() == (1, 3, 296, 221)
    assert torch.all(image <= 1)
    assert torch.all(image >= 0)

def test_im_file_resize():
    """Load and resize an image."""
    filename = Path(__file__).with_name("tree.jpg")
    expected_size = (123, 234)
    image = sumie.io.load_file(filename, size=expected_size)

    assert image.size() == (1, 3,) + expected_size
    assert torch.all(image <= 1)
    assert torch.all(image >= 0)

def test_im_url():
    """Load an image from a url."""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/606px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
    expected_size = (123, 234)
    image = sumie.io.load_url(url, size=expected_size)

    assert image.size() == (1, 3,) + expected_size
    assert torch.all(image <= 1)
    assert torch.all(image >= 0)

def test_save_1(tmpdir):
    """Save a tensor as an image"""
    data = torch.rand(1, 3, 240, 320)
    filename = str(tmpdir.join("folder", "test.jpg"))
    sumie.io.save(data, filename)
    
    assert os.path.isfile(filename)