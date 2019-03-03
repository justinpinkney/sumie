import sumie
import tests
import torch
from collections import namedtuple

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
