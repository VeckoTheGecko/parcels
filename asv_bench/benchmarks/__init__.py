import os


def _skip_slow():
    """
    Use this function to skip slow or highly demanding tests.

    Use it as a `Class.setup` method or a `function.setup` attribute.

    Examples
    --------
    >>> from . import _skip_slow
    >>> def time_something_slow():
    ...     pass
    ...
    >>> time_something.setup = _skip_slow
    """
    if os.environ.get("ASV_SKIP_SLOW", "0") == "1":
        raise NotImplementedError("Skipping this test...")
