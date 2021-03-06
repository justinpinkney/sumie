import torch

import sumie

class ModuleMonitor():
    """Stores the outputs for a module."""
    def __init__(self, module, pos=None):
        self.values = None
        self.hook_ref = module.register_forward_hook(self.hook)
        self.pos = pos

    def hook(self, module, hook_in, hook_out):
        if self.pos:
            val_shape = hook_out.shape
            selected = hook_out[:, :, int(self.pos[0]*val_shape[2]), int(self.pos[1]*val_shape[3])]
            self.values = selected[:,:,None,None]
        else:
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

    def __init__(self, module, channel, func=torch.mean, batch=0):
        self.channel = channel
        self.func = func
        self.monitor = ModuleMonitor(module)
        self.batch = batch

    @property
    def objective(self):
        if self.monitor.values is not None:
            target_channel = self.monitor.values[self.batch, self.channel, :, :]
            return self.func(target_channel)

    def __del__(self):
        self.monitor.remove()
        
class DeepDream():
    
    def __init__(self, module, func=torch.norm):
        self.monitor = ModuleMonitor(module)
        self.func = func
        
    @property
    def objective(self):
        if self.monitor.values is not None:
            return self.func(self.monitor.values)

def Content(image, model, module):
    """Make a TargetActivations objective with the given image."""
    monitor = ModuleMonitor(module)
    model(image)
    objective = TargetActivations(module, monitor.values.detach())
    monitor.remove()
    return objective
        
class TargetActivations():
    """Minimise the distance bewteen a module activation and some target."""

    def __init__(self, module, target, func=None, batch=0):
        self.monitor = ModuleMonitor(module)
        self.target = target
        self.criterion = torch.nn.MSELoss()
        self.func = func
        self.batch = batch

    @property
    def objective(self):
        value = self.monitor.values
        if value is not None:
            if self.func:
                value = self.func(value[self.batch, ...].unsqueeze(0))
            return -self.criterion(value, self.target)

class BatchMatchActivations():
    """Compares activations between two images in a batch"""

    def __init__(self, module, batch_idx_1, batch_idx_2, func=None):
        self.monitor = ModuleMonitor(module)
        self.batch_idx_1 = batch_idx_1
        self.batch_idx_2 = batch_idx_2
        self.criterion = torch.nn.MSELoss()
        self.func = func

    @property
    def objective(self):
        values = self.monitor.values
        if values is not None:
            value1 = values[self.batch_idx_1, ...]
            value2 = values[self.batch_idx_2, ...]
            if self.func:
                value1 = self.func(value1)
                value2 = self.func(value2)
            return -self.criterion(value1, value2)

class Style():
    def __init__(self, image, model, modules, weights=None, batch=0):
        self.weights = weights
        self.monitors = []
        self.objectives = []
        self.batch = batch
        
        for module in modules:
            self.monitors.append(ModuleMonitor(module))
            
        model(image)
        for monitor, module in zip(self.monitors, modules):
            target = sumie.utils.gram_matrix(monitor.values.detach())
            objective = TargetActivations(module, target, func=sumie.utils.gram_matrix, batch=self.batch)
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


class Direction():
    
    def __init__(self, module, target, pos=None):
        self.monitor = sumie.objectives.ModuleMonitor(module, pos)
        self.target = target.detach()
        
    @property
    def objective(self):
        if self.monitor.values is not None:
            picked_value = torch.mean(self.monitor.values, (2, 3)).squeeze(0)
            return direction_func(picked_value, 
                                  torch.mean(self.target, (2, 3)).squeeze(0))

    
def direction_func(x, y):
    cossim_pow = 2
    eps = 1e-4
    xy_dot = torch.dot(x, y)
    x_mag = torch.sqrt(torch.dot(x, x))
    y_mag = torch.sqrt(torch.dot(y, y))
    cossims = xy_dot / (eps + x_mag) / (eps + y_mag)
    #floored_cossims = torch.max(torch.Tensor((0.1,)), cossims)
    return torch.mean(xy_dot * cossims**cossim_pow)