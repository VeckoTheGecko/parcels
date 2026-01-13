#!/usr/bin/env python
import importlib
import sys

from numpydoc.validate import validate

PUBLIC_MODULES = [
    "scipy." + s
    for s in [
        "cluster",
        "cluster.vq",
        "cluster.hierarchy",
        "constants",
        "datasets",
        "differentiate",
        "fft",
        "fftpack",
        "integrate",
        "interpolate",
        "io",
        "io.arff",
        "io.matlab",
        "io.wavfile",
        "linalg",
        "linalg.blas",
        "linalg.cython_blas",
        "linalg.lapack",
        "linalg.cython_lapack",
        "linalg.interpolative",
        "ndimage",
        "odr",
        "optimize",
        "optimize.elementwise",
        "signal",
        "signal.windows",
        "sparse",
        "sparse.linalg",
        "sparse.csgraph",
        "spatial",
        "spatial.distance",
        "spatial.transform",
        "special",
        "stats",
        "stats.contingency",
        "stats.distributions",
        "stats.mstats",
        "stats.qmc",
        "stats.sampling",
    ]
]


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


def main():
    # load in numpydoc config
    to_check = []

    # get a list of all public objects
    for module in PUBLIC_MODULES:
        mod = importlib.import_module(module)
        try:
            to_check.extend(obj for f in mod.__all__ if (obj := f"{module}.{f}") not in PUBLIC_MODULES)
        except AttributeError:
            # needed for some deprecated modules
            continue

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
