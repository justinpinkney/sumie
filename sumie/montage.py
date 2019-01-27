from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import skimage

def load_image(filename, im_size):
    im = Image.open(filename).resize(im_size)
    original_im = np.asarray(im)
    target = torch.Tensor(original_im[:im_size[1],:im_size[0],:])
    return target.permute(2,1,0).unsqueeze(0)

def show(t):
    """Display the image tensor 1x3xhxw"""
    no_batch = t.detach().squeeze(0)
    im = no_batch.permute(2,1,0)/255
    plt.imshow(im)
    plt.show()
    
def make_filters(filename, im_size, filter_size):
    """Makes a bank of filters from image tiles."""
    im = Image.open(filename).resize(im_size)
    original_im = np.asarray(im)
    
    max_w_filters = math.floor(im_size[1]/filter_size)
    max_h_filters = math.floor(im_size[0]/filter_size)
    
    print(f"h = {max_h_filters}, w = {max_w_filters}")
    print(original_im.shape)
    
    filters = skimage.util.view_as_blocks(original_im[:max_w_filters*filter_size, 
                                                      :max_h_filters*filter_size, 
                                                      :], 
                                          (filter_size, filter_size, 3))

    filters = filters.reshape(-1,filter_size,filter_size,3)
    weights = torch.Tensor(filters).permute(0,3,1,2)
    return weights
