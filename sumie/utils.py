import torch

def max_to_avg_pool(model):
    """Swaps all MaxPool2d layers for AvgPool2d."""
    for name, child in model.named_children():
        if isinstance(child, torch.nn.MaxPool2d):
            new_child = make_avg_pool(child)
            model.__setattr__(name, new_child)
        else:
            max_to_avg_pool(child)

def make_avg_pool(max):
    avg = torch.nn.AvgPool2d(kernel_size=max.kernel_size,
                             stride=max.stride,
                             padding=max.padding,
                             )
    return avg

def remove_inplace(model):
    """Change all inplace modules to non-inplace version."""
    for child in model.modules():
        if hasattr(child, 'inplace'):
            child.inplace = False


def gram_matrix(input):
    """Compute the gram matrix of an input."""
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)