import torch

import sumie

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

    def __init__(self, module, target, func=None):
        self.monitor = ModuleMonitor(module)
        self.target = target
        self.criterion = torch.nn.MSELoss()
        self.func = func

    @property
    def objective(self):
        value = self.monitor.values
        if value is not None:
            if self.func:
                value = self.func(value)
            return -self.criterion(value, self.target)
        
class Style():
    def __init__(self, image, model, modules, weights=None):
        self.weights = weights
        self.monitors = []
        self.objectives = []
        
        for module in modules:
            self.monitors.append(ModuleMonitor(module))
            
        model(image)
        for monitor, module in zip(self.monitors, modules):
            target = sumie.utils.gram_matrix(monitor.values.detach())
            objective = TargetActivations(module, target, func=sumie.utils.gram_matrix)
            self.objectives.append(objective)
        
        for monitor in self.monitors:
            monitor.remove()

    @property
    def objective(self):
        value = 0
        for objective in self.objectives:
            if objective.objective is None:
                return None
            
            value += objective.objective
        return value
    
class White():
    
    def __init__(self, image):
        self.monitor = ModuleMonitor(image)
        
    @property
    def objective(self):
        if self.monitor.values is not None:
            return (self.monitor.values - 1).mean()