import sumie
import torch
import pytest
import tests

from pathlib import Path

@pytest.mark.parametrize("param", ["rgb", "fft"])
@pytest.mark.parametrize("decorrelate", [True, False])
def test_image_init(param, decorrelate):
    """Allow initialisation with an existing image."""
    filename = Path(__file__).with_name("tree.jpg")
    init_image = sumie.io.load_file(filename, size=(224, 224))

    image = sumie.Image(param=param, decorrelate=decorrelate, init=init_image)
    
    output_image = image.get_image()
    difference = torch.abs(output_image - init_image)
    assert torch.nn.functional.mse_loss(output_image, init_image) < 1e-3

def test_optimiser():
    """Optimiser class optimises an image given an objective."""
    model = tests.test_objectives.make_net()
    objective = sumie.objectives.ConvChannel(model[0], 0)
    image = sumie.Image(10)
    original_image = image.get_image()

    model(image())
    start_value = objective.objective

    opt = sumie.Optimiser()
    opt.run(image, model, objective, iterations=10)

    new_image = image.get_image()
    assert torch.any(new_image != original_image)

    model(image())
    end_value = objective.objective

    assert end_value > start_value
