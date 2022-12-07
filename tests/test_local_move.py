from functools import partial

import numpy as np
import pytest
from jax import grad, jit
from jax import numpy as jnp
from jax import vmap

from timemachine.ff import Forcefield
from timemachine.integrator import VelocityVerletIntegrator
from timemachine.lib import LangevinIntegrator, custom_ops
from timemachine.md.enhanced import get_solvent_phase_system
from timemachine.md.local_resampling import local_resampling_move
from timemachine.potentials.jax_utils import delta_r
from timemachine.testsystems.ligands import get_biphenyl


def make_hmc_mover(x, logpdf_fxn, dt=0.1, n_steps=100):
    masses = np.ones(len(x))

    def force_fxn(x):
        return -grad(logpdf_fxn)(x)

    integrator = VelocityVerletIntegrator(force_fxn, masses=masses, dt=dt)

    def augmented_logpdf(x, v):
        return logpdf_fxn(x) - np.sum(0.5 * masses[:, np.newaxis] * v ** 2)

    @jit
    def augmented_proposal(x0, v0):
        logp_before = augmented_logpdf(x0, v0)
        x1, v1 = integrator._update_via_fori_loop(x0, v0, n_steps=n_steps)
        logp_after = augmented_logpdf(x1, v1)

        log_accept_prob = jnp.clip(jnp.nan_to_num(logp_after - logp_before, nan=-np.inf), a_max=0.0)

        return (x1, v1), log_accept_prob

    def hmc_move(x0):
        v0 = np.random.randn(*x.shape)
        (x1, v1), log_accept_prob = augmented_proposal(x0, v0)

        x_new = x1 if np.random.rand() < jnp.exp(log_accept_prob) else x0
        return x_new, log_accept_prob

    return hmc_move


def expect_no_drift(x0, move_fxn, observable_fxn, n_local_resampling_iterations=100):
    traj = [jnp.array(x0)]
    aux_traj = []

    for _ in range(n_local_resampling_iterations):
        updated, aux = move_fxn(traj[-1])

        traj.append(updated)
        aux_traj.append(aux)

    expected_selection_fraction_traj = np.array([observable_fxn(x) for x in traj])

    # TODO: don't hard-code T, thresholds, etc.
    T = 10
    assert n_local_resampling_iterations > 2 * T
    avg_at_start = np.mean(expected_selection_fraction_traj[:T])
    avg_at_end = np.mean(expected_selection_fraction_traj[-T:])

    ratio = avg_at_end / avg_at_start
    deviated_by_50percent_or_more = ratio <= 0.5 or ratio >= 1.5
    if deviated_by_50percent_or_more:
        msg = f"""
            observable avg over start frames = {avg_at_start:.3f}
            observable avg over end frames = {avg_at_end:.3f}
            but averages of this (and all other observables) should be constant over time
        """
        raise RuntimeError(msg)

    return traj, aux_traj


def naive_local_resampling_move(
    x,
    target_logpdf_fxn,
    particle_selection_log_prob_fxn,
    mcmc_move,
):
    """WARNING: deliberately incorrect, with a key step ablated for testing purposes!

    local_resampling_move, but with restraint potential disabled"""
    n_particles = len(x)

    # select particles to be updated
    selection_probs = np.exp(particle_selection_log_prob_fxn(x))
    assert np.min(selection_probs) >= 0 and np.max(selection_probs) <= 1, "selection_probs must be in [0,1]"
    assert selection_probs.shape == (n_particles,), "must compute per-particle selection_probs"
    selection_mask = np.random.rand(n_particles) < selection_probs  # TODO: factor out dependence on global numpy rng?

    # NOTE: missing restraint! will result in incorrect sampling

    # def restrained_logpdf_fxn(x) -> float:
    #    log_p_i = particle_selection_log_prob_fxn(x)
    #    return target_logpdf_fxn(x) + bernoulli_logpdf(log_p_i, selection_mask)

    def subproblem_logpdf(x_sub) -> float:
        x_full = x.at[selection_mask].set(x_sub)
        return target_logpdf_fxn(x_full)  # return restrained_logpdf_fxn(x_full)

    # apply any valid MCMC move to this subproblem
    x_sub = x[selection_mask]
    x_next_sub, aux = mcmc_move(x_sub, subproblem_logpdf)
    x_next = x.at[selection_mask].set(x_next_sub)

    return x_next, aux


@pytest.mark.nightly(reason="Slow")
def test_ideal_gas():
    """Run HMC on subsets of an ideal gas system, where the subsets are selected based on a geometric criterion.
    Assert that an observable based on this criterion doesn't drift after a large number of updates when using
    local_resampling_move, and assert that it does drift when using an ablated version of local_resampling_move."""
    np.random.seed(2022)

    # make 2D ideal gas system
    box_size = 5
    dim = 2

    box = np.eye(dim) * box_size
    n_particles = 1000

    def ideal_gas_2d_logpdf_fxn(x):
        return 0.0

    x0 = np.random.rand(n_particles, 2) * box_size

    # make function that preferentially selects particles near center
    center = np.ones(dim) * (box_size / 2)
    r0 = box_size / 6

    def central_particle_selection_log_prob_fxn(x):
        distance_from_center = lambda x_i: jnp.linalg.norm(delta_r(x_i, center, box))
        r = vmap(distance_from_center)(x)

        return jnp.where(r > r0, -10 * (r - r0) ** 4, 0.0)

    # define any correct MCMC move fxn -- e.g. a composition of other correct MCMC moves
    def run_multiple_hmc_moves(x, logpdf_fxn, dt=1e-4, n_steps=100, n_moves=100):
        hmc_move = make_hmc_mover(x, logpdf_fxn, dt=dt, n_steps=n_steps)
        log_accept_probs = []
        for _ in range(n_moves):
            x, log_accept_prob = hmc_move(x)
            log_accept_probs.append(log_accept_prob)

        return x, np.array(log_accept_probs)

    # define correct and incorrect version of local move
    # (with the same target, same particle selection method, and same MCMC move)
    common_kwargs = dict(
        target_logpdf_fxn=ideal_gas_2d_logpdf_fxn,
        particle_selection_log_prob_fxn=central_particle_selection_log_prob_fxn,
        mcmc_move=run_multiple_hmc_moves,
    )
    correct_local_move = partial(local_resampling_move, **common_kwargs)
    incorrect_local_move = partial(naive_local_resampling_move, **common_kwargs)

    # test that num particles near center doesn't change dramatically
    def particle_frac_near_center(x):
        return np.mean(np.exp(central_particle_selection_log_prob_fxn(x)))

    def assert_correctness(local_move):
        # primary assertion: expect no drift in % of particles in resampled region
        traj, aux_traj = expect_no_drift(
            x0, local_move, observable_fxn=particle_frac_near_center, n_local_resampling_iterations=100
        )

        # secondary assertion: confirm move was not trivial
        avg_accept_prob = np.mean([np.mean(np.exp(log_accept_probs)) for log_accept_probs in aux_traj])
        assert avg_accept_prob > 0.1, "traj was probably trivial: didn't accept enough moves"
        assert np.max(np.abs(traj[0] - traj[-1])) > r0, "traj was probably trivial: didn't move very far"
        initially_within_region = vmap(lambda x_i: jnp.linalg.norm(delta_r(x_i, center, box)))(x0) < r0
        all_moved = ((traj[-1] - traj[0])[initially_within_region] != 0).all()
        assert all_moved, "traj was probably trivial: didn't update all particles in local region"

    # expect local move to be correct
    assert_correctness(correct_local_move)

    # expect failure with ablated version of local move
    with pytest.raises(RuntimeError):
        assert_correctness(incorrect_local_move)


def test_local_md_particle_density():
    """Verify that the average particle density around a single particle is stable.

    In the naive implementation of local md, a vacuum can appear around the local idxs. See naive_local_resampling_move
    for what the incorrect implementation looks like. The vacuume is introduced due to discretization error where in a step
    a particle moves away from the local idxs and is frozen in the next round of local MD.
    """
    mol, _ = get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_1_1_0_sc.py")

    temperature = 300
    dt = 1.5e-3
    friction = 0.0
    seed = 2022
    cutoff = 1.2

    # Have to minimize, else there can be clashes and the local moves will cause crashes
    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(mol, ff, 0.0)

    local_idxs = np.array([len(coords) - 1], dtype=np.int32)

    nblist = custom_ops.Neighborlist_f32(coords.shape[0])

    # Construct list of atoms in the inner shell
    nblist.set_row_idxs(local_idxs.astype(np.uint32))

    intg = LangevinIntegrator(temperature, dt, friction, masses, seed)

    intg_impl = intg.impl()

    v0 = np.zeros_like(coords)
    bps = []
    for p, bp in zip(sys_params, unbound_potentials):
        bps.append(bp.bind(p).bound_impl(np.float32))

    def observable(x):
        ixns = nblist.get_nblist(coords, box, cutoff)
        flattened = np.concatenate(ixns).reshape(-1)
        inner_shell_idxs = np.unique(flattened)

        # Combine all of the indices that are involved in the inner shell
        subsystem_idxs = np.unique(np.concatenate([inner_shell_idxs, local_idxs]))
        return len(subsystem_idxs)

    ctxt = custom_ops.Context(coords, v0, box, intg_impl, bps)

    def local_move(x, steps=1):
        xs, boxes = ctxt.multiple_steps_local(steps, local_idxs)
        return xs, boxes

    expect_no_drift(coords, local_move, observable, n_local_resampling_iterations=50)
