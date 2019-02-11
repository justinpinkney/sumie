import sumie.sumie
import torch
import pytest

from pathlib import Path

@pytest.mark.parametrize("param", ["rgb", "fft"])
def test_image_init(param):
    """Allow initialisation with an existing image."""
    filename = Path(__file__).with_name("tree.jpg")
    init_image = sumie.io.load_file(filename, size=(224, 224))

    image = sumie.sumie.Image(param=param, init=init_image)
    
    output_image = image.get_image()
    difference = torch.abs(output_image - init_image)
    assert torch.nn.functional.mse_loss(output_image, init_image) < 1e-3
