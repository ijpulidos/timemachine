import os
from typing import Union

import numpy as np
from openmm import Vec3, app, unit

from timemachine.ff import sanitize_water_ff


def strip_units(coords):
    return np.array(coords.value_in_unit_system(unit.md_unit_system))


def build_protein_system(host_pdbfile: Union[app.PDBFile, str], protein_ff: str, water_ff: str, phosaa_ff: str = "amber/phosaa10"):
    """
    Build a solvated protein system.

    Parameters
    ----------
    host_pdbfile : Union[app.PDBFile, str]
        The PDB file or path to the PDB file of the host protein.
    protein_ff : str
        The force field parameters for the protein.
    water_ff : str
        The force field parameters for water.
    phosaa_ff : str, optional
        The force field parameters for phosphorous amino acids (default is "amber/phosaa10").
        Note that "amber/phosaa10.xml" requires `openmmforcefields` to be installed in the env.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - solvated_host_system : openmm.System
            The solvated protein system.
        - solvated_host_coords : numpy.ndarray
            The coordinates of the solvated protein system.
        - box : numpy.ndarray
            The box dimensions of the solvated protein system.
        - topology : openmm.Topology
            The topology of the solvated protein system.
        - nwa : int
            The number of water atoms in the solvated protein system.

    Raises
    ------
    TypeError
        If host_pdbfile is neither a string nor an openmm PDBFile object.

    Notes
    -----
    This function builds a solvated protein system using the provided PDB file and force field parameters.
    It adds solvent molecules around the protein using the specified force field parameters for protein,
    water, and optionally phosphorous amino acids.

    Example
    -------
    solvated_host_system, solvated_host_coords, box, topology, nwa = build_protein_system(
        "host.pdb", "amber14/protein.ff14SB", "amber14/tip3p", phosaa_ff="amber/phosaa10")
    """
    host_ff = app.ForceField(f"{protein_ff}.xml", f"{water_ff}.xml", f"{phosaa_ff}.xml")
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    padding = 1.0
    box_lengths = np.amax(host_coords, axis=0) - np.amin(host_coords, axis=0)

    box_lengths = box_lengths + padding
    box = np.eye(3, dtype=np.float64) * box_lengths

    modeller.addSolvent(
        host_ff, boxSize=np.diag(box) * unit.nanometers, neutralize=False, model=sanitize_water_ff(water_ff)
    )
    solvated_host_coords = strip_units(modeller.positions)

    nha = host_coords.shape[0]
    nwa = solvated_host_coords.shape[0] - nha

    print("building a protein system with", nha, "protein atoms and", nwa, "water atoms")
    solvated_host_system = host_ff.createSystem(
        modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )

    return solvated_host_system, solvated_host_coords, box, modeller.topology, nwa


def build_water_system(box_width, water_ff: str):
    ff = app.ForceField(f"{water_ff}.xml")

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    m = app.Modeller(top, pos)

    boxSize = Vec3(box_width, box_width, box_width) * unit.nanometers
    m.addSolvent(ff, boxSize=boxSize, model=sanitize_water_ff(water_ff))

    system = ff.createSystem(m.getTopology(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)

    positions = m.getPositions()
    positions = strip_units(positions)

    assert m.getTopology().getNumAtoms() == positions.shape[0]

    # TODO: minimize the water box (BFGS or scipy.optimize)
    return system, positions, np.eye(3) * box_width, m.getTopology()
