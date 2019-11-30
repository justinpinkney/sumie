import sumie
import torch
import pytest
import tests

from pathlib import Path
from glob import glob

@pytest.mark.parametrize("param", ["rgb", "fft", "pyramid"])
@pytest.mark.parametrize("decorrelate", [True, False])
def test_image_init(param, decorrelate):
    """Allow initialisation with an existing image."""
    filename = Path(__file__).with_name("tree.jpg")
    init_image = sumie.io.load_file(filename, size=(224, 224))

    image = sumie.Image(param=param, decorrelate=decorrelate, init=init_image)
    
    output_image = image.get_image()
    difference = torch.abs(output_image - init_image)
    assert torch.nn.functional.mse_loss(output_image, init_image) < 1e-3

@pytest.mark.slow
def test_image_init_cppn():
    """Use lower freq image to test cppn"""
    filename = Path(__file__).with_name("tree_lf.jpg")
    init_image = sumie.io.load_file(filename, size=(50, 50))

    image = sumie.Image(param='cppn', size=50, init=init_image)
    
    output_image = image.get_image()
    difference = torch.abs(output_image - init_image)
    assert torch.nn.functional.mse_loss(output_image, init_image) < 1e-3
    
def test_optimiser(simple_net):
    """Optimiser class optimises an image given an objective."""
    model, objective, image = setup_optimiser(simple_net)
    original_image, start_value = get_state(model, image, objective)
    
    opt = sumie.Optimiser(image, model, objective)
    opt.run(iterations=10)
    
    new_image, end_value = get_state(model, image, objective)
    assert torch.any(new_image != original_image)
    assert end_value > start_value

def test_optimiser_history(simple_net):
    """Optimiser stores objective history."""
    model, objective, image = setup_optimiser(simple_net)
    n_iterations = 10
    
    opt = sumie.Optimiser(image, model, objective)
    opt.run(iterations=10)
    new_image, end_value = get_state(model, image, objective)

    # +1 as we get the initial and final losses
    assert len(opt.history) == n_iterations + 1
    assert opt.history[-1] == end_value

def test_optimiser_output(tmpdir, simple_net):
    """Save image per iteration of the optimiser."""
    model, objective, image = setup_optimiser(simple_net)
    n_iterations = 10
    
    opt = sumie.Optimiser(image, model, objective)
    opt.run(iterations=10, output=Path(tmpdir))

    search = str(tmpdir.join('*.jpg'))
    assert len(glob(search)) == 11
    
def test_optimiser_output_str(tmpdir, simple_net):
    """Save image to new folder of type string."""
    model, objective, image = setup_optimiser(simple_net)
    n_iterations = 10
    new_folder = 'tmp'
    
    opt = sumie.Optimiser(image, model, objective)
    opt.run(iterations=10, output=new_folder)

    search = new_folder + '/*.jpg'
    assert len(glob(search)) == 11

# Utility functions
def get_state(model, image, objective):
    """Return the current image and value of objective."""
    model(image())
    return image.get_image(), objective.objective

def setup_optimiser(model):
    objective = sumie.objectives.ConvChannel(model[0], 0)
    image = sumie.Image(10)
    
    return model, objective, image
