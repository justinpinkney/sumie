import sumie.objectives

import torch

def test_hook_add_remove():
    """Objective can remove its hook."""
    model = make_net()
    objective = sumie.objectives.ConvChannel(model[0], 1)
    assert len(model[0]._forward_hooks) == 1
    objective.remove()
    assert len(model[0]._forward_hooks) == 0

def test_conv_objective():
    """Conv objective gets mean value of channel by default."""
    model = make_net()
    objective = sumie.objectives.ConvChannel(model[0], 0)
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    model(input)
    assert objective.objective.data == 1/16

def test_conv_custom_objective():
    """Allow changing of the reduction function of conv channel"""
    model = make_net()
    objective = sumie.objectives.ConvChannel(model[0], 0, torch.max)
    input = torch.zeros(1, 3, 4, 4)
    input[0, 0, 0, 0] = 1
    model(input)
    assert objective.objective.data == 1

def make_net():
    model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 3, bias=False, padding=1),
            torch.nn.Conv2d(6, 1, 1, bias=False, padding=1),
            )
    torch.nn.init.constant_(model[0].weight, 0)
    torch.nn.init.constant_(model[1].weight, 1)
    model[0].weight.data[0,0,0,0] = 1
    return model
