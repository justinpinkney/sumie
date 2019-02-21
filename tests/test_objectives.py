import sumie
import tests
import torch

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

