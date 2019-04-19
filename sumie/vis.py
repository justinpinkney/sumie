import torch
import matplotlib.pyplot as plt

def show(t, figsize=(5,5)):
    """Display the image tensor 1x3xhxw.
    
    Image values are expected to be in the range 0->1.
    """
    
    no_batch = t.detach().cpu().squeeze(0)
    im = no_batch.permute(1, 2, 0)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    plt.axis('off')
    ax.grid(False)
    plt.imshow(im)
    plt.show()
    