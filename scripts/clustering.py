import sumie
import numpy as np
from sklearn.cluster import KMeans
import torchvision
import torch

class Direction():
    
    def __init__(self, module, target):
        self.monitor = sumie.objectives.ModuleMonitor(module)
        self.target = target.detach()
        
    @property
    def objective(self):
        if self.monitor.values is not None:
            return direction_func(torch.mean(self.monitor.values, (2, 3)).squeeze(0), 
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

def change_scale(opt, i):
    opt.image.transforms[-2].factor *= 1.0045

if __name__ == '__main__':
    model = torchvision.models.densenet121(pretrained=True).eval()
    modules = list(model.modules())
    sumie.utils.remove_inplace(model)
    #url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Hyams_beach.jpg/640px-Hyams_beach.jpg'
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Gothic_Chapel_Peterhof_tonemapped.jpg/640px-Gothic_Chapel_Peterhof_tonemapped.jpg'

    device = 'cuda'
    model.to(device)
    base_image = sumie.io.load_url(url)
    base_image = sumie.utils.normalise(base_image)

    selected_module = modules[187]
    monitor = sumie.objectives.ModuleMonitor(selected_module)
    model(base_image.to(device))
    all_target = monitor.values

    values = all_target.detach().squeeze(0).cpu().numpy()
    values = values.reshape((values.shape[0], -1)).T
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(values)
    print(kmeans.cluster_centers_.shape)

    for i in range(n_clusters):
        cluster = torch.Tensor(kmeans.cluster_centers_[i, :])
        cluster = cluster[None, :, None, None]
        im = sumie.Image((1000, 300), param='fft', transforms=[
                            sumie.transforms.PositionJitter(8),
                            sumie.transforms.ScaleJitter(1.01),
                            sumie.transforms.RotationJitter(0.1),
                            sumie.transforms.PositionJitter(8),
                            sumie.transforms.Interpolate(0.1),
                            sumie.transforms.Normalise(),
                      ])

        im.to(device)

        content = Direction(selected_module, cluster.to(device))

        opt = sumie.Optimiser(im, model, content)
        opt.add_callback(change_scale)
        opt.run(iterations=512, lr=0.05, progress=True)
        sumie.io.save(im.get_image(), f'tmp/output_{i:03}.png')
