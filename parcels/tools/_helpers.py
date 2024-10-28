"""Internal helpers for Parcels."""

from __future__ import annotations

import functools
import textwrap
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parcels import Field, FieldSet, ParticleSet

PACKAGE = "Parcels"


def deprecated(msg: str = "") -> Callable:
    """Decorator marking a function as being deprecated

    Parameters
    ----------
    msg : str, optional
        Custom message to append to the deprecation warning.

    Examples
    --------
    ```
    @deprecated("Please use `another_function` instead")
    def some_old_function(x, y):
        return x + y

    @deprecated()
    def some_other_old_function(x, y):
        return x + y
    ```
    """
    if msg:
        msg = " " + msg

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg_formatted = (
                f"`{func.__qualname__}` is deprecated and will be removed in a future release of {PACKAGE}.{msg}"
            )

            warnings.warn(msg_formatted, category=DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)

        patch_docstring(wrapper, f"\n\n.. deprecated:: {msg}")
        return wrapper

    return decorator


def deprecated_made_private(func: Callable) -> Callable:
    return deprecated(
        "It has moved to the internal API as it is not expected to be directly used by "
        "the end-user. If you feel that you use this code directly in your scripts, please "
        "comment on our tracking issue at https://github.com/OceanParcels/Parcels/issues/1695.",
    )(func)


def patch_docstring(obj: Callable, extra: str) -> None:
    obj.__doc__ = f"{obj.__doc__ or ''}{extra}".strip()


def field_repr(field: Field) -> str:
    """Return a pretty repr for Field"""
    out = f"""<{type(field).__name__}>
    name            : {field.name!r}
    grid            : {field.grid!r}
    extrapolate time: {field.allow_time_extrapolation!r}
    time_periodic   : {field.time_periodic!r}
    gridindexingtype: {field.gridindexingtype!r}
    to_write        : {field.to_write!r}
"""
    return textwrap.dedent(out).strip()


def _format_list_items_multiline(items: list[str], level: int = 1) -> str:
    """Given a list of strings, formats them across multiple lines.

    Uses indentation levels of 4 spaces provided by ``level``.

    Example
    -------
    >>> output = _format_list_items_multiline(["item1", "item2", "item3"], 4)
    >>> f"my_items: {output}"
    my_items: [
        item1,
        item2,
        item3,
    ]
    """
    if len(items) == 0:
        return "[]"

    assert level >= 1, "Indentation level >=1 supported"
    indentation_str = level * 4 * " "
    indentation_str_end = (level - 1) * 4 * " "

    items_str = ",\n".join([textwrap.indent(i, indentation_str) for i in items])
    return f"[\n{items_str}\n{indentation_str_end}]"


def particleset_repr(pset: ParticleSet) -> str:
    """Return a pretty repr for ParticleSet"""
    if len(pset) < 10:
        particles = [repr(p) for p in pset]
    else:
        particles = [repr(pset[i]) for i in range(7)] + ["..."]

    out = f"""<{type(pset).__name__}>
    fieldset   : {pset.fieldset}
    pclass     : {pset.pclass}
    repeatdt   : {pset.repeatdt}
    # particles: {len(pset)}
    particles  : {_format_list_items_multiline(particles)}
"""
    return textwrap.dedent(out).strip()


def fieldset_repr(fieldset: FieldSet) -> str:
    """Return a pretty repr for FieldSet"""
    fields_repr = "\n".join([repr(f) for f in fieldset.get_fields()])

    out = f"""<{type(fieldset).__name__}>
    fields:
{textwrap.indent(fields_repr, 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def default_repr(obj: Any):
    return object.__repr__(obj)
