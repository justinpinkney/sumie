import sumie
import torch
import torchvision

model = torchvision.models.vgg16(pretrained=True).eval()
sumie.utils.remove_inplace(model)

def normalise(image):
    
    mean = torch.as_tensor([0.485, 0.456, 0.406])
    std = torch.as_tensor([0.229, 0.224, 0.225])
    return (image - mean[None,:,None,None]) /std[None,:,None,None]
 
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
    opt.image.transforms[-2].factor *= 1.01

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Piper_betle_plant.jpg/320px-Piper_betle_plant.jpg'
imsize = 512

device = 'cuda'
model.to(device)
base_image = sumie.io.load_url(url, size=(imsize, imsize))
base_image = normalise(base_image)

for selected_module in model.features.children():
    monitor = sumie.objectives.ModuleMonitor(selected_module)
    model.features(base_image.to(device))
    target = monitor.values

    im = sumie.Image(imsize, param='fft', transforms=[
                        sumie.transforms.PositionJitter(8),
                        sumie.transforms.ScaleJitter(1.01),
                        sumie.transforms.RotationJitter(1),
                        sumie.transforms.PositionJitter(8),
                        sumie.transforms.Interpolate(0.1),
                        sumie.transforms.Normalise(),
                  ])

    im.to(device)

    content = Direction(selected_module, target.detach())

    opt = sumie.Optimiser(im, model.features, content)
    opt.add_callback(change_scale)
    opt.run(iterations=256, lr=0.05, progress=True)
    sumie.io.save(im.get_image(), f'tmp/output_{selected_module}.png')


