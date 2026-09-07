"""``torch``, imported the first time something reaches for it.

The numpy decoders - :func:`~rankfield.margin`, :func:`~rankfield.deficit`,
:func:`~rankfield.probabilities`, the levels table - and the store reader need no torch, so
importing ``rankfield`` must not need it either: torch is the ``torch`` extra. The modules
that do use it bind this proxy in place of the module. It imports on the first attribute
read and, when there is nothing to import, names the extra that supplies it.

Annotations are postponed everywhere in this package (``from __future__ import
annotations``), so a ``torch.Tensor`` in a signature is a string and never reaches here.
"""
from __future__ import annotations

import functools
import importlib
from types import ModuleType

HINT = ('torch is not installed. It is the "torch" extra: pip install "rankfield[torch]". '
        "encode(), decode_groups(), to_device() and the restore need it; margin(), deficit(), "
        "probabilities() and the store reader do not.")


class _Torch(ModuleType):
    """Stands in for ``torch`` until an attribute is read, then becomes it."""

    def __init__(self) -> None:
        super().__init__("torch")

    def __getattr__(self, name: str):
        try:
            real = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(HINT) from exc
        value = getattr(real, name)
        setattr(self, name, value)      # bound on this proxy: __getattr__ will not run for it again
        return value


torch = _Torch()


def have_torch() -> bool:
    """Whether torch can be imported - what a GPU backend asks before claiming it is available."""
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError:
        return False
    return True


def no_grad(fn):
    """``torch.no_grad()`` entered when the function runs, not when its module is imported -
    ``@torch.no_grad()`` as a decorator would resolve the proxy at import time."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper
