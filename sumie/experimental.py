import torch
import sumie

def SmoothDownsample(downFolds):
    
    # TODO check validity
    
    transforms = []
    first = True
    downFolds = int(downFolds)
    
    while downFolds / 2 >= 1:
        transforms.append(sumie.transforms.Interpolate(0.5))
        if first:
            transforms.append(torch.nn.AvgPool2d(downFolds, stride=1, padding=downFolds//4))
            first = False
        else:
            transforms.append(torch.nn.AvgPool2d(downFolds, stride=1, padding=downFolds//2))
        downFolds //= 2
        
    return torch.nn.Sequential(*transforms)

class CentreDuplicate(torch.nn.Module):
    """Downscales image and superimposes back in the centre."""

    def __init__(self, factor):
        super(CentreDuplicate, self).__init__()
        self.factor = factor
        
    def forward(self, image):
        # TODO should we detach?
        rescaled_image = torch.nn.functional.interpolate(image, scale_factor=self.factor)
        _, _, h, w = image.shape
        _, _, new_h, new_w = rescaled_image.shape
        matched_image = torch.nn.ConstantPad2d(round((h-new_h)/2), 0)(rescaled_image)
        mask = torch.ones(image.shape).to(image.device)
        mask[:,:,
             round((h-new_h)/2) : round((h+new_h)/2),
             round((w-new_w)/2) : round((w+new_w)/2)] = 0
        new_image = mask*image + (1-mask)*matched_image
        return new_image