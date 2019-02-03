import torch
import sumie.transforms

def test_position_jitter_size():
    """Get the same size out as in."""

    transform = sumie.transforms.PositionJitter(1)
    image = torch.randn(1, 3, 10, 10)
    out = transform(image)

    assert out.size() == image.size()

def test_position_jitter_axis():
    """Jitter only operates on x,y axes."""

    transform = sumie.transforms.PositionJitter(10)
    image = torch.randn(1, 3, 1, 1)
    out = transform(image)

    assert torch.all(out == image)

def test_random_crop_size():
    """Get expected output size."""

    size = (10, 12)
    transform = sumie.transforms.RandomCrop(size)
    image = torch.randn(1, 3, 23, 42)
    out = transform(image)

    assert out.size() == (1, 3) + size

def test_interpolate_size():
    """Interpolate should change size by desired factor."""

    size = (10, 12)
    factors = (0.5, 1, 2)

    for factor in factors:
        expected_size = tuple(factor*x for x in size)
        transform = sumie.transforms.Interpolate(factor)
        image = torch.randn(1, 3, *size)
        out = transform(image)

        assert out.size() == (1, 3) + expected_size
