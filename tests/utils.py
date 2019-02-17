import torch

def make_net(weights='fixed'):
    model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 3, bias=False, padding=1),
            torch.nn.Conv2d(6, 1, 1, bias=False, padding=1),
            )
    
    if weights == 'fixed':
        torch.nn.init.constant_(model[0].weight, 0)
        torch.nn.init.constant_(model[1].weight, 1)
        model[0].weight.data[0,0,0,0] = 1
    
    return model