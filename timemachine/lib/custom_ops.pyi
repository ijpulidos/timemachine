from typing import List, Optional

import numpy

class BoundPotential:
    def __init__(self, potential: Potential, params: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def execute(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], lam: float) -> tuple: ...
    def execute_fixed(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], lam: float) -> numpy.typing.NDArray[numpy.uint64]: ...
    def get_potential(self) -> Potential: ...
    def size(self) -> int: ...

class CentroidRestraint_f32(Potential):
    def __init__(self, arg0: numpy.typing.NDArray[numpy.int32], arg1: numpy.typing.NDArray[numpy.int32], arg2: float, arg3: float) -> None: ...

class CentroidRestraint_f64(Potential):
    def __init__(self, arg0: numpy.typing.NDArray[numpy.int32], arg1: numpy.typing.NDArray[numpy.int32], arg2: float, arg3: float) -> None: ...

class ChiralAtomRestraint_f32(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class ChiralAtomRestraint_f64(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class ChiralBondRestraint_f32(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32], signs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class ChiralBondRestraint_f64(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32], signs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class Context:
    def __init__(self, x0: numpy.typing.NDArray[numpy.float64], v0: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], integrator: Integrator, bps: List[BoundPotential], barostat: Optional[MonteCarloBarostat] = ...) -> None: ...
    def finalize(self, lamb: float) -> None: ...
    def get_box(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def get_v_t(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def get_x_t(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def initialize(self, lamb: float) -> None: ...
    def multiple_steps(self, lambda_schedule: numpy.typing.NDArray[numpy.float64], store_du_dl_interval: int = ..., store_x_interval: int = ...) -> tuple: ...
    def multiple_steps_U(self, lamb: float, n_steps: int, lambda_windows: numpy.typing.NDArray[numpy.float64], store_u_interval: int, store_x_interval: int) -> tuple: ...
    def set_v_t(self, velocities: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def set_x_t(self, coords: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def step(self, lamb: float) -> None: ...

class FanoutSummedPotential(Potential):
    def __init__(self, potentials: List[Potential]) -> None: ...
    def get_potentials(self) -> List[Potential]: ...

class FlatBottomBond_f32(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class FlatBottomBond_f64(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicAngle_f32(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32], lamb_mult: Optional[numpy.typing.NDArray[numpy.int32]] = ..., lamb_offset: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...

class HarmonicAngle_f64(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32], lamb_mult: Optional[numpy.typing.NDArray[numpy.int32]] = ..., lamb_offset: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...

class HarmonicBond_f32(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32], lamb_mult: Optional[numpy.typing.NDArray[numpy.int32]] = ..., lamb_offset: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...

class HarmonicBond_f64(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32], lamb_mult: Optional[numpy.typing.NDArray[numpy.int32]] = ..., lamb_offset: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...

class Integrator:
    def __init__(self, *args, **kwargs) -> None: ...

class LangevinIntegrator(Integrator):
    def __init__(self, masses: numpy.typing.NDArray[numpy.float64], temperature: float, dt: float, friction: float, seed: int) -> None: ...

class MonteCarloBarostat:
    def __init__(self, arg0: int, arg1: float, arg2: float, arg3: List[List[int]], arg4: int, arg5, arg6: int) -> None: ...
    def get_interval(self) -> int: ...
    def set_interval(self, arg0: int) -> None: ...
    def set_pressure(self, arg0: float) -> None: ...

class Neighborlist_f32:
    def __init__(self, N: int) -> None: ...
    def compute_block_bounds(self, arg0: numpy.typing.NDArray[numpy.float64], arg1: numpy.typing.NDArray[numpy.float64], arg2: int) -> tuple: ...
    def get_nblist(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], cutoff: float) -> List[List[int]]: ...
    def reset_row_idxs(self) -> None: ...
    def resize(self, size: int) -> None: ...
    def set_row_idxs(self, idxs: numpy.typing.NDArray[numpy.uint32]) -> None: ...

class Neighborlist_f64:
    def __init__(self, N: int) -> None: ...
    def compute_block_bounds(self, arg0: numpy.typing.NDArray[numpy.float64], arg1: numpy.typing.NDArray[numpy.float64], arg2: int) -> tuple: ...
    def get_nblist(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], cutoff: float) -> List[List[int]]: ...
    def reset_row_idxs(self) -> None: ...
    def resize(self, size: int) -> None: ...
    def set_row_idxs(self, idxs: numpy.typing.NDArray[numpy.uint32]) -> None: ...

class NonbondedAllPairs_f32(Potential):
    def __init__(self, lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float, atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedAllPairs_f32_interpolated(Potential):
    def __init__(self, lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float, atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedAllPairs_f64(Potential):
    def __init__(self, lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float, atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedAllPairs_f64_interpolated(Potential):
    def __init__(self, lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float, atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedInteractionGroup_f32(Potential):
    def __init__(self, row_atom_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedInteractionGroup_f32_interpolated(Potential):
    def __init__(self, row_atom_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedInteractionGroup_f64(Potential):
    def __init__(self, row_atom_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedInteractionGroup_f64_interpolated(Potential):
    def __init__(self, row_atom_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...
    def disable_hilbert_sort(self) -> None: ...
    def set_nblist_padding(self, arg0: float) -> None: ...

class NonbondedPairListPrecomputed_f32(Potential):
    def __init__(self, pair_idxs: numpy.typing.NDArray[numpy.int32], w_offsets: numpy.typing.NDArray[numpy.float64], beta: float, cutoff: float) -> None: ...

class NonbondedPairListPrecomputed_f64(Potential):
    def __init__(self, pair_idxs: numpy.typing.NDArray[numpy.int32], w_offsets: numpy.typing.NDArray[numpy.float64], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f32(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f32_interpolated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f32_negated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f32_negated_interpolated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f64(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f64_interpolated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f64_negated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f64_negated_interpolated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], lambda_plane_idxs_i: numpy.typing.NDArray[numpy.int32], lambda_offset_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class PeriodicTorsion_f32(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32], lamb_mult: Optional[numpy.typing.NDArray[numpy.int32]] = ..., lamb_offset: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...

class PeriodicTorsion_f64(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32], lamb_mult: Optional[numpy.typing.NDArray[numpy.int32]] = ..., lamb_offset: Optional[numpy.typing.NDArray[numpy.int32]] = ...) -> None: ...

class Potential:
    def __init__(self, *args, **kwargs) -> None: ...
    def execute(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], lam: float) -> tuple: ...
    def execute_du_dx(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], lam: float) -> numpy.typing.NDArray[numpy.float64]: ...
    def execute_selective(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], lam: float, compute_du_dx: bool, compute_du_dp: bool, compute_du_dl: bool, compute_u: bool) -> tuple: ...
    def execute_selective_batch(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], boxes: numpy.typing.NDArray[numpy.float64], lambs: numpy.typing.NDArray[numpy.float64], compute_du_dx: bool, compute_du_dp: bool, compute_du_dl: bool, compute_u: bool) -> tuple: ...

class SummedPotential(Potential):
    def __init__(self, potentials: List[Potential], params_sizes: List[int]) -> None: ...
    def get_potentials(self) -> List[Potential]: ...

class VelocityVerletIntegrator(Integrator):
    def __init__(self, dt: float, cbs: numpy.typing.NDArray[numpy.float64]) -> None: ...

def cuda_device_reset() -> None: ...
def rmsd_align(arg0: numpy.typing.NDArray[numpy.float64], arg1: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]: ...
