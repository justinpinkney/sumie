import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from pathlib import Path

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

def save(data, filename):
    """Save a tensor image as a file.
    TODO If the tensor has a batch then save as multiple files.
    """
    if data.shape[0] > 1:
        raise NotImplementedError("Batch saving not yet implemented.")
        
    if isinstance(filename, str):
         filename = Path(filename)
            
    if not filename.parent.is_dir():
        filename.parent.mkdir()
            
    data_no_batch = data.cpu().detach().squeeze(0)
    image_data = np.uint8(data_no_batch.permute(2,1,0).numpy()*255)
    image = Image.fromarray(image_data)
    image.save(filename)
    