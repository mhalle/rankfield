"""GPU paths for the restore: ``metal`` (Apple, torch.mps.compile_shader) and ``triton_gpu``
(CUDA). Both expose ``available()`` and ``run_ranked(ranks, support, out, tables, lut, *,
levels, clip, paint)`` and make the torch path's decisions bit for bit."""
