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
    
    def __init__(self, module, channel, func=torch.mean):
        self.objective = None
        self.channel = channel
        self.hook_ref = module.register_forward_hook(self.hook)
        self.func = func
        self.values = None

    def hook(self, module, hook_in, hook_out):
        target = hook_out.clone()[0, self.channel, :, :]
        self.objective = self.func(target)
        self.values = hook_out.clone()

    def __del__(self):
        self.hook_ref.remove()
        

class Style():

    def __init__(self, module, model, image):
        self.hook_ref = module.register_forward_hook(self.init_hook)
        model(image)
        self.hook_ref.remove()
        self.hook_ref = module.register_forward_hook(self.hook)
        self.values = []
        self.criterion = torch.nn.MSELoss()

    def init_hook(self, module, hook_in, hook_out):
        target = hook_out.detach()
        self.target = self.gram_matrix(target)

    def hook(self, module, hook_in, hook_out):
        output = hook_out.clone()
        self.objective = self.criterion(self.gram_matrix(output), self.target)

    def __del__(self):
        self.hook_ref.remove()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)
