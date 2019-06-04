import numpy as np

from rdkit import Chem

from system import serialize
from system import forcefield
from openforcefield.typing.engines.smirnoff import ForceField

host_potentials, host_conf, (host_params, host_param_groups), host_masses = serialize.deserialize_system('examples/host_acd.xml')

test_sdf = """@<TRIPOS>MOLECULE
CD Set 1, Guest ID 1: 1-butylamine
   17    16     1     0     0
SMALL
No Charge or Current Charge


@<TRIPOS>ATOM
      1 N1          19.6330    18.8470    24.1380 N.4        7 MOL      -0.305700
      2 H1          19.6610    17.8160    24.1890 H          7 MOL       0.312600
      3 H2          18.6400    19.1270    24.1700 H          7 MOL       0.312600
      4 HN11        20.0780    19.2050    24.9980 H          7 MOL       0.312600
      5 C1          20.3230    19.3660    22.8950 C.3        7 MOL       0.045900
      6 H3          21.3710    19.0570    22.9510 H          7 MOL       0.091600
      7 H4          20.2800    20.4580    22.9300 H          7 MOL       0.091600
      8 C2          19.6960    18.8570    21.5830 C.3        7 MOL      -0.055000
      9 H5          19.7300    17.7650    21.5600 H          7 MOL       0.042900
     10 H6          18.6450    19.1580    21.5390 H          7 MOL       0.042900
     11 C3          20.4470    19.4250    20.3620 C.3        7 MOL       0.006600
     12 H7          21.4980    19.1260    20.4030 H          7 MOL       0.031100
     13 H8          20.4140    20.5170    20.3820 H          7 MOL       0.031100
     14 C4          19.8300    18.9250    19.0420 C.3        7 MOL      -0.088600
     15 H9          19.8750    17.8340    18.9840 H          7 MOL       0.042600
     16 H10         20.3740    19.3360    18.1900 H          7 MOL       0.042600
     17 H11         18.7840    19.2340    18.9640 H          7 MOL       0.042600
@<TRIPOS>BOND
     1     1     2 1   
     2     1     3 1   
     3     1     4 1   
     4     1     5 1   
     5     5     6 1   
     6     5     7 1   
     7     5     8 1   
     8     8     9 1   
     9     8    10 1   
    10     8    11 1   
    11    11    12 1   
    12    11    13 1   
    13    11    14 1   
    14    14    15 1   
    15    14    16 1   
    16    14    17 1   
@<TRIPOS>SUBSTRUCTURE
1 MOL 1 TEMP 0 **** **** 0 ROOT"""

mol = Chem.MolFromMol2Block(test_sdf, sanitize=True, removeHs=False, cleanupSubstructures=True)
smirnoff = ForceField("test_forcefields/smirnoff99Frosst.offxml")

guest_potentials, guest_params, guest_param_groups, guest_conf, guest_masses = forcefield.parameterize(mol, smirnoff)

combined_potentials = forcefield.combiner(
    host_potentials, guest_potentials,
    host_params, guest_params,
    host_param_groups, guest_param_groups,
    host_conf, guest_conf,
    host_masses, guest_masses)


c = mol.GetConformer(0)
conf = np.array(c.GetPositions(), dtype=np.float64)

print(conf)
print(potentials)
