import torch

class ModuleMonitor():
    """Stores the outputs for a module."""
    def __init__(self, module):
        self.values = None
        self.hook_ref = module.register_forward_hook(self.hook)

    def hook(self, module, hook_in, hook_out):
        self.values = hook_out

    def remove(self):
        self.hook_ref.remove()

class Composite():
    """Adds objectives with optional weighting."""

    def __init__(self, children, weights=None):
        self.children = children
        if weights:
            self.weights = weights
        else:
            self.weights = [1 for el in self.children]
        
    @property
    def objective(self):
        total = 0
        for child, weight in zip(self.children, self.weights):
            total += weight*child.objective
        return total

class Linear():

    def __init__(self, module, index):
        self.index = index
        self.monitor = ModuleMonitor(module)

    @property
    def objective(self):
        if self.monitor.values is not None:
            return self.monitor.values[0, self.index]

    def __del__(self):
        self.monitor.remove()


class ConvChannel():

    def __init__(self, module, channel, func=torch.mean):
        self.channel = channel
        self.func = func
        self.monitor = ModuleMonitor(module)

    @property
    def objective(self):
        if self.monitor.values is not None:
            target_channel = self.monitor.values[0, self.channel, :, :]
            return self.func(target_channel)

    def __del__(self):
        self.monitor.remove()
        
class DeepDream():
    
    def __init__(self, module):
        self.monitor = ModuleMonitor(module)
        
    @property
    def objective(self):
        if self.monitor.values is not None:
            return self.monitor.values.norm()

def Content(image, model, module):
    """Make a TargetActivations objective with the given image."""
    monitor = ModuleMonitor(module)
    model(image)
    objective = TargetActivations(module, monitor.values.detach())
    monitor.remove()
    return objective
        
class TargetActivations():
    """Minimise the distance bewteen a module activation and some target."""

    def __init__(self, module, target):
        self.monitor = ModuleMonitor(module)
        self.target = target
        self.criterion = torch.nn.MSELoss()

    @property
    def objective(self):
        if self.monitor.values is not None:
            return -self.criterion(self.monitor.values, self.target)
        
class Style():

    def __init__(self, modules, model, image, weights=None):
        hook_ref = dict()
        self.target = dict()
        self.weights = weights
        self.objective = 0
        for module in modules:
            hook_ref[module] = module.register_forward_hook(self.init_hook)
        model(image)
        for module in modules:
            hook_ref[module].remove()

        self.hook_ref = dict()
        for idx, module in enumerate(modules):
            self.hook_ref[module] = module.register_forward_hook(lambda a, b, c, this_idx=idx: self.hook(a, b, c, this_idx))

        self.criterion = torch.nn.MSELoss()

    def init_hook(self, module, hook_in, hook_out):
        target = hook_out.detach()
        self.target[module] = self.gram_matrix(target)

    def hook(self, module, hook_in, hook_out, idx):
        output = hook_out
        if self.weights:
            weight = self.weights[idx]
        else:
            weight = 1
        if idx == 0:
            self.objective = 0
        this_contribution = weight*self.criterion(self.gram_matrix(output), self.target[module])
        self.objective += this_contribution

    def __del__(self):
        for hook_ref in self.hook_ref.values():
            hook_ref.remove()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

class White():
    
    def __init__(self, image):
        self.monitor = ModuleMonitor(image)
        
    @property
    def objective(self):
        if self.monitor.values is not None:
            value = (self.monitor.values - 1).norm()
            return -1*value