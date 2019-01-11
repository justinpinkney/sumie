import torch

class Linear():

    def __init__(self, module, index):
        self.objective = None
        self.index = index
        self.hook_ref = module.register_forward_hook(self.hook)

    def hook(self, module, hook_in, hook_out):
        self.objective = hook_out[0, self.index]

    def __del__(self):
        self.hook_ref.remove()
        

class ConvChannel():
    
    def __init__(self, module, channel):
        self.objective = None
        self.channel = channel
        self.hook_ref = module.register_forward_hook(self.hook)

    def hook(self, module, hook_in, hook_out):
        self.objective = torch.mean(hook_out[0, self.channel, :, :])

    def __del__(self):
        self.hook_ref.remove()
