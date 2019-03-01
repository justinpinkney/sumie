import sumie
import tests
import torch
import pytest

def test_hook_add_remove():
    """Objective can remove its hook."""
    model = tests.utils.make_net()
    objective = sumie.objectives.ConvChannel(model[0], 1)
    assert len(model[0]._forward_hooks) == 1
    del objective
    assert len(model[0]._forward_hooks) == 0

def test_conv_objective():
    """Conv objective gets mean value of channel by default."""
    model = tests.utils.make_net()
    objective = sumie.objectives.ConvChannel(model[0], 0)
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    model(input)
    assert objective.objective.data == 1/16

def test_conv_custom_objective():
    """Allow changing of the reduction function of conv channel"""
    model = tests.utils.make_net()
    objective = sumie.objectives.ConvChannel(model[0], 0, torch.max)
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    model(input)
    assert objective.objective.data == 1

def test_deep_dream():
    """Deep dream objective should be the l2 norm of module."""
    model = tests.utils.make_net()
    objective = sumie.objectives.DeepDream(model[0])
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    model(input)
    assert objective.objective.data == 1
    
    input[0, 0, 0, 0] = 2
    input[0, 0, 2, 2] = 1
    model(input)
    assert objective.objective.data ** 2 == 5
    
def test_white():
    """Objective to encourage the image to be white."""
    image = sumie.inputs.RgbImage(10, noise=0)
    objective = sumie.objectives.White(image)
    image()
    expected = (10*10*3*1) ** 0.5
    assert pytest.approx(objective.objective.item(), 1e-6) == -expected