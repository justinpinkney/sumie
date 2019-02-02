import torch
import sumie.transforms

def test_scale():

    transform = sumie.transforms.PositionJitter(1)
    image = torch.randn(1, 3, 10, 10)
    out = transform(image)

    assert out.size() == image.size()
