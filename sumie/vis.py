import torch
import matplotlib.pyplot as plt

def show(t, figsize=(5,5)):
    """Display the image tensor 1x3xhxw.
    
    Image values are expected to be in the range 0->1.
    """
    im = im_to_np(t)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    plt.axis('off')
    ax.grid(False)
    plt.imshow(im)
    plt.show()
    
def im_to_np(im):
    """Convert image tensor 1x3xhxw to something suitable for matlplotlib"""
    
    no_batch = im.detach().cpu().squeeze(0)
    return no_batch.permute(1, 2, 0)
    
