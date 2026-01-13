#!/usr/bin/env python
import functools
import importlib
import sys
import types

from numpydoc.validate import validate

PUBLIC_MODULES = ["parcels", "parcels.interpolators"]

skip_errors = [
    "GL01",
    "GL02",
    "GL03",
    "GL05",
    "GL07",
    "GL09",
    "SS02",
    "SS03",
    "SS05",
    "SS06",
    "ES01",
    "PR01",
    "PR02",
    "PR03",
    "PR04",
    "PR06",
    "PR07",
    "PR08",
    "PR09",
    "PR10",
    "RT01",
    "RT02",
    "RT03",
    "RT04",
    "RT05",
    "YD01",
    "SA01",
    "SA02",
    "SA03",
    "SA04",
    "EX01",  # remove when gh-7168 is resolved
]


def is_built_in(type_or_instance: type | object):
    if isinstance(type_or_instance, type):
        return type_or_instance.__module__ == "builtins"
    else:
        return type_or_instance.__class__.__module__ == "builtins"


def walk_module(module_str: str, public_api: list[str] | None = None) -> list[str]:
    if public_api is None:
        public_api = []

    module = importlib.import_module(module_str)
    try:
        all_ = module.__all__
    except AttributeError:
        print(f"No __all__ variable found in public module {module_str!r}")
        return public_api

    if module_str not in public_api:
        try:
            _ = module.__doc__
            public_api.append(module_str)
        except AttributeError:
            pass  # module has no docstring
    for item_str in all_:
        item = getattr(module, item_str)
        if isinstance(item, types.ModuleType):
            walk_module(f"{module_str}.{item_str}", public_api)
        if isinstance(item, (types.FunctionType,)):
            public_api.append(f"{module_str}.{item_str}")
        elif is_built_in(item):
            print(f"Found builtin at '{module_str}.{item_str}' of type {type(item)}")
            continue
        elif isinstance(item, type):
            public_api.append(f"{module_str}.{item_str}")
            walk_class(module_str, item, public_api)
        else:
            print(
                f"Encountered unexpected public object at '{module_str}.{item_str}' of {item!r} in public API. Don't know how to handle with numpydoc - ignoring."
            )

    return public_api


def get_public_class_attrs(class_: type) -> set[str]:
    return {a for a in dir(class_) if not a.startswith("_")}


def walk_class(module_str: str, class_: type, public_api: list[str]) -> list[str]:
    class_str = class_.__name__

    # attributes that were introduced by this class specifically - not from inheritance
    attrs = get_public_class_attrs(class_) - functools.reduce(
        set.add, (get_public_class_attrs(base) for base in class_.__bases__)
    )

    for attr_str in attrs:
        attr = getattr(class_, attr_str)
        try:
            _ = attr.__doc__
            public_api.append(f"{module_str}.{class_str}.{attr_str}")
        except AttributeError:
            pass  # attribute doesn't have a docstring
    return public_api


def main():
    public_api = []
    for module in PUBLIC_MODULES:
        public_api += walk_module(module)

    public_api = filter(lambda x: x != "parcels", public_api)  # For some reason doesn't work on root parcels package?
    errors = 0
    for item in public_api:
        try:
            res = validate(item)
        except AttributeError:
            continue
        if res["type"] in ("module", "float", "int", "dict"):
            continue
        for err in res["errors"]:
            if err[0] not in skip_errors:
                print(f"{item}: {err}")
                errors += 1
    sys.exit(errors)


if __name__ == "__main__":
    main()
