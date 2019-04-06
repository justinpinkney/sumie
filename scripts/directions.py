import sumie
import torch
import torchvision

model = torchvision.models.densenet121(pretrained=True).eval()
sumie.utils.remove_inplace(model)

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

def run(target, selected_module, average=True):
    device = 'cuda'

    im = sumie.Image(imsize, param='fft', transforms=[
                        sumie.transforms.PositionJitter(8),
                        sumie.transforms.ScaleJitter(1.01),
                        sumie.transforms.RotationJitter(0.1),
                        sumie.transforms.PositionJitter(8),
                        sumie.transforms.Interpolate(0.1),
                        sumie.transforms.Normalise(),
                  ])

    im.to(device)

    content = Direction(selected_module, target.detach())

    opt = sumie.Optimiser(im, model, content)
    opt.add_callback(change_scale)
    opt.run(iterations=512, lr=0.05, progress=True)
    return im.get_image()

if __name__ == '__main__':
    #url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Piper_betle_plant.jpg/320px-Piper_betle_plant.jpg'
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Hyams_beach.jpg/640px-Hyams_beach.jpg'
    imsize = 256
    average = False

    device = 'cuda'
    model.to(device)
    base_image = sumie.io.load_url(url, size=(imsize, imsize))
    base_image = sumie.utils.normalise(base_image)

    for i, selected_module in enumerate(model.modules()):
        if i != 187:
            continue
        try:

            monitor = sumie.objectives.ModuleMonitor(selected_module)
            model(base_image.to(device))
            all_target = monitor.values
            if average:
                im_out = run(all_target, selected_module)
                sumie.io.save(im_out, f'tmp/output_{i:03}.png')
            else:
                size = all_target.shape
                for x in range(size[2]):
                    for y in range(size[3]):
                        target = all_target[0,:,x,y]
                        target = target[None,:,None,None]
                        im_out = run(target, selected_module)
                        sumie.io.save(im_out, f'tmp/output_{i:03}_{x:03}_{y:03}.png')

        except Exception as e:
            print(f'failed on {selected_module}')
            print(e)

