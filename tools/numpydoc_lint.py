#!/usr/bin/env python
import importlib
import sys
from pprint import pprint
from types import ModuleType

from numpydoc.validate import validate

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
encountered_types = set()


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
        public_api.append(module_str)
    for item_str in all_:
        item = getattr(module, item_str)
        encountered_types.add(type(item))
        if isinstance(item, ModuleType):
            walk_module(f"{module_str}.{item_str}", public_api)
        elif isinstance(item, type):
            public_api.append(f"{module_str}.{item_str}")
            walk_class(module_str, item, public_api)
            # ... # Handle a custom type
        else:
            public_api.append(f"{module_str}.{item_str}")

    return public_api


def walk_class(module_str: str, class_: type, public_api: list[str]) -> list[str]:
    class_str = class_.__name__
    attrs = [a for a in dir(class_) if not a.startswith("_")]
    for attr_str in attrs:
        attr = getattr(class_, attr_str)
        try:
            _ = attr.__doc__
            public_api.append(f"{module_str}.{class_str}.{attr_str}")
        except AttributeError:
            pass  # attribute doesn't have a docstring
    return public_api


def main():
    # load in numpydoc config
    pprint(walk_module("parcels"))
    pprint(encountered_types)
    return
    errors = 0
    for item in to_check:
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
