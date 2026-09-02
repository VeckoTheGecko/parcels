import numpy as np
from re_assert import Matches

from parcels import Particle, ParticleFile, ParticleSet


def test_particlefile_repr(tmp_parquet):
    pfile_repr = repr(ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s")))
    match = Matches(
        r"""\<ParticleFile\>
    path                : .*
    outputdt            : 1.0
    metadata            : .*""",
    )
    match.assert_matches(pfile_repr)
    # assert_simple_repr(ParticleFile, kwargs)


def test_field_repr(fieldset):
    Matches(r"Field\(name=.*, model=.*\)").assert_matches(repr(fieldset.U))


def test_vectorfield_repr(fieldset):
    Matches(r"\<.*VectorField object at.*\>").assert_matches(repr(fieldset.UV))


def test_xgrid_repr(fieldset):
    Matches(r"\<.*XGrid object at.*\>").assert_matches(repr(fieldset.U.grid))


def test_fieldset_repr(fieldset):
    Matches(r"\<.*FieldSet object at.*\>").assert_matches(repr(fieldset))


def test_particleset_repr(fieldset):
    Matches(r"\<.*ParticleSet object at.*\>").assert_matches(repr(ParticleSet(fieldset, pclass=Particle)))
