import sumie
import torch

def test_normalise():
    data = torch.zeros(1, 3, 5, 5)
    result = sumie.utils.normalise(data)
    assert torch.all(result < data)
