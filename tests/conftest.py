import numpy as np
import pytest
import torch


def logits(K=8, shape=(6, 10, 12), seed=0, noise=0.0):
    """Blobby, spatially correlated logits - confident interiors AND near-tie boundaries."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
    out = []
    for k in range(K):
        c = torch.rand(3, generator=g) * torch.tensor(shape, dtype=torch.float32)
        d = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).sqrt()
        out.append(6.0 - 0.9 * d + (noise * torch.randn(shape, generator=g) if noise else 0.0))
    return torch.stack(out)


@pytest.fixture
def lg():
    return logits()


def devices():
    from rankfield.backends import metal
    out = ["cpu"]
    if torch.backends.mps.is_available() and metal.available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


__all__ = ["logits", "devices", "np"]
