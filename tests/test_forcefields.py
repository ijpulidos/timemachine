import os
from glob import glob
from pathlib import Path
from warnings import catch_warnings

import numpy as np
import pytest
from common import load_split_forcefields, temporary_working_dir

from timemachine import constants
from timemachine.ff import Forcefield, combine_params
from timemachine.ff.handlers.deserialize import deserialize_handlers

pytestmark = [pytest.mark.nocuda]


def test_serialization_of_ffs():
    for path in glob("timemachine/ff/params/smirnoff_*.py"):
        handlers, protein_ff, water_ff = deserialize_handlers(open(path).read())
        ff = Forcefield.from_handlers(handlers, protein_ff=protein_ff, water_ff=water_ff)
        assert ff.protein_ff == constants.DEFAULT_PROTEIN_FF
        assert ff.water_ff == constants.DEFAULT_WATER_FF
        for handle in ff.get_ordered_handles():
            assert handle is not None, f"{path} failed to deserialize correctly"


def test_loading_forcefield_from_file():
    builtin_ffs = glob("timemachine/ff/params/smirnoff_*.py")
    for path in builtin_ffs:
        # Use the full path
        ff = Forcefield.load_from_file(path)
        assert ff is not None
        # Use full path as Path object
        ff = Forcefield.load_from_file(Path(path))
        assert ff is not None
        with catch_warnings(record=True) as w:
            # Load using just file name of the built in
            ff = Forcefield.load_from_file(os.path.basename(path))
            assert ff is not None
        assert len(w) == 0

    for prefix in ["", "timemachine/ff/params/"]:
        path = os.path.join(prefix, "nosuchfile.py")
        with pytest.raises(ValueError) as e:
            Forcefield.load_from_file(path)
        assert path in str(e.value)
        with pytest.raises(ValueError):
            Forcefield.load_from_file(Path(path))
        assert path in str(e.value)

    with temporary_working_dir() as tempdir:
        # Verify that if a local file shadows a builtin
        for path in builtin_ffs:
            basename = os.path.basename(path)
            with open(basename, "w") as ofs:
                ofs.write("junk")
            with catch_warnings(record=True) as w:
                Forcefield.load_from_file(basename)
            assert len(w) == 1
            assert basename in str(w[0].message)
        with catch_warnings(record=True) as w:
            bad_ff = Path(tempdir, "jut.py")
            assert bad_ff.is_absolute(), "Must be absolute to cover test case"
            with open(bad_ff, "w") as ofs:
                ofs.write("{}")
            with pytest.raises(ValueError, match="Unsupported charge handler"):
                Forcefield.load_from_file(bad_ff)
        assert len(w) == 0


def test_load_default():
    """assert that load_default() is an alias for load_from_file(DEFAULT_FF)"""

    ref = Forcefield.load_from_file(constants.DEFAULT_FF)
    test = Forcefield.load_default()

    # ref == test evaluates to false, so assert equality of smirks lists and parameter arrays manually

    ref_handles = ref.get_ordered_handles()
    test_handles = test.get_ordered_handles()

    assert len(ref_handles) == len(test_handles)

    for ref_handle, test_handle in zip(ref.get_ordered_handles(), test.get_ordered_handles()):
        assert ref_handle.smirks == test_handle.smirks
        np.testing.assert_array_equal(ref_handle.params, test_handle.params)


def test_split():
    ffs = load_split_forcefields()

    def check(ff):
        params = ff.get_params()
        np.testing.assert_array_equal(ff.hb_handle.params, params.hb_params)
        np.testing.assert_array_equal(ff.ha_handle.params, params.ha_params)
        np.testing.assert_array_equal(ff.pt_handle.params, params.pt_params)
        np.testing.assert_array_equal(ff.it_handle.params, params.it_params)
        np.testing.assert_array_equal(ff.q_handle.params, params.q_params)
        np.testing.assert_array_equal(ff.q_handle_intra.params, params.q_params_intra)
        np.testing.assert_array_equal(ff.q_handle_solv.params, params.q_params_solv)
        np.testing.assert_array_equal(ff.lj_handle.params, params.lj_params)
        np.testing.assert_array_equal(ff.lj_handle_intra.params, params.lj_params_intra)
        np.testing.assert_array_equal(ff.lj_handle_solv.params, params.lj_params_solv)

        assert ff.get_ordered_handles() == [
            ff.hb_handle,
            ff.ha_handle,
            ff.pt_handle,
            ff.it_handle,
            ff.q_handle,
            ff.q_handle_intra,
            ff.q_handle_solv,
            ff.lj_handle,
            ff.lj_handle_intra,
            ff.lj_handle_solv,
        ]

        combined = combine_params(ff.get_params(), ff.get_params())
        np.testing.assert_array_equal((ff.hb_handle.params, ff.hb_handle.params), combined.hb_params)
        np.testing.assert_array_equal((ff.ha_handle.params, ff.ha_handle.params), combined.ha_params)
        np.testing.assert_array_equal((ff.pt_handle.params, ff.pt_handle.params), combined.pt_params)
        np.testing.assert_array_equal((ff.it_handle.params, ff.it_handle.params), combined.it_params)
        np.testing.assert_array_equal((ff.q_handle.params, ff.q_handle.params), combined.q_params)
        np.testing.assert_array_equal((ff.q_handle_intra.params, ff.q_handle_intra.params), combined.q_params_intra)
        np.testing.assert_array_equal((ff.q_handle_solv.params, ff.q_handle_solv.params), combined.q_params_solv)
        np.testing.assert_array_equal((ff.lj_handle.params, ff.lj_handle.params), combined.lj_params)
        np.testing.assert_array_equal((ff.lj_handle_intra.params, ff.lj_handle_intra.params), combined.lj_params_intra)
        np.testing.assert_array_equal((ff.lj_handle_solv.params, ff.lj_handle_solv.params), combined.lj_params_solv)

    check(ffs.ref)
    check(ffs.intra)
    check(ffs.solv)
    check(ffs.prot)
    check(ffs.prot)
