"""GPU paths for the restore: ``metal`` (Apple, torch.mps.compile_shader) and ``triton_gpu``
(CUDA). Both expose ``available()`` and ``run_ranked(ranks, support, out, tables, lut, *,
levels, clip, paint)`` and make the torch path's decisions bit for bit."""
from .._torch import torch


def max_class(ranks) -> int:
    """The largest ``class + 1`` stored in ``ranks``, for the label-table bounds check.

    Not ``int(ranks.max())``: torch has no max reduction for uint16 on ANY of the three
    backends - ``max_reduction_ushort_ushort`` on MPS, ``max_all not implemented for UInt16``
    on the CPU, ``max_all_cuda not implemented for UInt16`` on CUDA - and uint16 is exactly
    the dtype a store with more than 254 classes uses, so the bounds check took the kernel
    down before it ran, on the only stores that need it. Widening the whole array to int32 would fix that and cost a gigabyte on a
    whole-body part, so widen a chunk at a time and keep the temporary bounded.
    """
    if ranks.numel() == 0:
        return 0
    if ranks.dtype == torch.uint8:
        return int(ranks.max())
    flat = ranks.reshape(-1)
    step = 1 << 24
    return max(int(flat[i:i + step].to(torch.int32).max()) for i in range(0, flat.numel(), step))
