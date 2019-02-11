import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO


def load_file(filename, size=None):
    """Load an image from file as a tensor"""
    im = Image.open(filename)
    if size:
        im = im.resize(size)
    original_im = np.asarray(im)
    target = torch.Tensor(original_im)/255
    return target.permute(2,1,0).unsqueeze(0)

def load_url(url, size=None):
    """Load an image from url."""
    response = requests.get(url)
    contents = BytesIO(response.content)
    return load_file(contents, size=size)
