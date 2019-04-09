import sumie
import tests
import torch
import pytest
from collections import namedtuple

def test_hook_add_remove(simple_net):
    """Objective can remove its hook."""
    objective = sumie.objectives.ConvChannel(simple_net[0], 1)
    assert len(simple_net[0]._forward_hooks) == 1
    del objective
    assert len(simple_net[0]._forward_hooks) == 0

def test_conv_objective(simple_net):
    """Conv objective gets mean value of channel by default."""
    objective = sumie.objectives.ConvChannel(simple_net[0], 0)
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    simple_net(input)
    assert objective.objective.data == 1/16

def test_conv_custom_objective(simple_net):
    """Allow changing of the reduction function of conv channel"""
    objective = sumie.objectives.ConvChannel(simple_net[0], 0, torch.max)
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    simple_net(input)
    assert objective.objective.data == 1

def test_deep_dream(simple_net):
    """Deep dream objective should be the l2 norm of module."""
    objective = sumie.objectives.DeepDream(simple_net[0])
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    simple_net(input)
    assert objective.objective.data == 1
    
    input[0, 0, 0, 0] = 2
    input[0, 0, 2, 2] = 1
    simple_net(input)
    assert objective.objective.data ** 2 == 5
    
def test_white():
    """Objective to encourage the image to be white."""
    image = sumie.inputs.RgbImage(10, noise=0)
    objective = sumie.objectives.White(image)
    image()
    expected = 1
    assert pytest.approx(objective.objective.item(), 1e-6) == -expected

def test_composite():
    """Composite objective should add objectives with weights default to 1."""
    child_objective = namedtuple('child_objective', ['objective'])
    child1 = child_objective(1)
    child2 = child_objective(3)
    objective = sumie.objectives.Composite((child1, child2))
    assert objective.objective == 4

def test_composite_weights():
    """Composite objective should add objectives with weights."""
    child_objective = namedtuple('child_objective', ['objective'])
    child1 = child_objective(1)
    child2 = child_objective(3)
    weights = [1/10, 1/30]
    objective = sumie.objectives.Composite((child1, child2), weights=weights)
    assert objective.objective == 2/10
    
def test_content(simple_net):
    """Content loss aims to reproduce the activations exactly"""
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    
    objective = sumie.objectives.Content(input, simple_net, simple_net[1])
    
    simple_net(input)
    assert objective.objective.data == 0
    
    input[0, 0, 0, 0] = 2
    input[0, 0, 2, 2] = 1
    simple_net(input)
    assert objective.objective.data == -2/6/6
    
def test_style_one_module(simple_net):
    """Style loss is mse of gram matrix for target"""
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    
    objective = sumie.objectives.Style(input, simple_net, [simple_net[1],])
    
    val = simple_net(input)
    gram1 = sumie.utils.gram_matrix(val)
    assert objective.objective.data == 0
    
    input[0, 0, 0, 0] = 2
    input[0, 0, 2, 2] = 1
    val = simple_net(input)
    gram2 = sumie.utils.gram_matrix(val)
    expected = (gram2 - gram1) ** 2
    assert objective.objective.item() == -expected.item()
    
def test_style_multiple_modules(simple_net):
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    
    objective = sumie.objectives.Style(input, simple_net, simple_net[0:2])
    
    simple_net(input)
    assert objective.objective.data == 0
    
    # TODO test multiple with actual values
    