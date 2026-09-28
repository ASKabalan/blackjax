# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Algorithms to adapt the MCLMC kernel parameters, namely step size and L."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree
from blackjax.util import generate_unit_vector, incremental_value_update, pytree_size


class MCLMCAdaptationState(NamedTuple):
    """Represents the tunable parameters for MCLMC adaptation.

    L
        The momentum decoherent rate for the MCLMC algorithm.
    step_size
        The step size used for the MCLMC algorithm.
    inverse_mass_matrix
        A matrix used for preconditioning.
    """

    L: float
    step_size: float
    inverse_mass_matrix: float


def mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    logdensity_fn=None,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    diagonal_preconditioning=True,
    params=None,
    l_factor=0.4,
    progress_bar=False,
    print_rate=None,
):
    """
    Finds the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function built by ``mclmc.build_kernel``.  Its call signature
        must be ``kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L,
        step_size)``, matching the standard BlackJAX kernel pattern.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    logdensity_fn
        The log-density function of the target distribution.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    desired_energy_var
        The desired energy variance for the MCMC algorithm.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.
    diagonal_preconditioning
        Whether to do diagonal preconditioning (i.e. a mass matrix)
    params
        Initial params to start tuning from (optional)
    l_factor
        The factor scaling the estimated autocorrelation length to obtain momentum decoherence length L.
    print_rate
        The rate at which the progress bar is updated. If None, defaults to
        printing every `num_steps // 20` steps.

    Returns
    -------
    final_state
        The final integrator state after the three tuning phases.
    final_params
        An ``MCLMCAdaptationState`` containing the adapted ``L``,
        ``step_size``, and ``inverse_mass_matrix``.
    total_num_tuning_integrator_steps
        The total number of integrator steps consumed across all three
        tuning phases (frac_tune1 + frac_tune2 + frac_tune3 of
        ``num_steps``).

    Example
    -------
    .. code-block:: python

        kernel = blackjax.mcmc.mclmc.build_kernel(integrator=integrator)

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            num_tuning_steps,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=logdensity_fn,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=preconditioning,
        )
    """
    if logdensity_fn is None:
        raise ValueError(
            "logdensity_fn is required. Pass the log-density function of the "
            "target distribution."
        )

    dim = pytree_size(state.position)
    if params is None:
        params = MCLMCAdaptationState(
            jnp.sqrt(dim), jnp.sqrt(dim) * 0.25, inverse_mass_matrix=jnp.ones((dim,))
        )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    total_num_tuning_integrator_steps = 0

    num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
        num_steps * frac_tune2
    )
    num_steps2 += diagonal_preconditioning * (num_steps2 // 3)
    num_steps3 = round(num_steps * frac_tune3)

    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        logdensity_fn=logdensity_fn,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        desired_energy_var=desired_energy_var,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        diagonal_preconditioning=diagonal_preconditioning,
        progress_bar=progress_bar,
        print_rate=print_rate,
    )(state, params, num_steps, part1_key)
    total_num_tuning_integrator_steps += num_steps1 + num_steps2

    if num_steps3 >= 2:
        state, params = make_adaptation_L(
            mclmc_kernel,
            logdensity_fn,
            frac=frac_tune3,
            l_factor=l_factor,
            progress_bar=progress_bar,
            print_rate=print_rate,
        )(state, params, num_steps, part2_key)
        total_num_tuning_integrator_steps += num_steps3

    return state, params, total_num_tuning_integrator_steps


class MCLMCTuningState(NamedTuple):
    """State of a chunked MCLMC tuning.

    step
        Number of tuning steps done, over the three stages.
    state
        The MCLMC chain state.
    params
        The current ``MCLMCAdaptationState``.
    adaptive_state
        ``(time, x_average, step_size_max)`` of the step-size adaptation.
    streaming_avg
        ``(weight, [E[x], E[x^2]])`` of the posterior-size estimate.
    samples
        Flattened positions of the L stage, one row per step.
    """

    step: Array
    state: ArrayLikeTree
    params: MCLMCAdaptationState
    adaptive_state: tuple
    streaming_avg: tuple
    samples: Array


class ChunkedMCLMCAdaptation(NamedTuple):
    """``init``, ``run_chunk`` and ``final`` of a chunked MCLMC tuning, and
    ``num_steps``, the total number of tuning steps over its stages."""

    init: Callable
    run_chunk: Callable
    final: Callable
    num_steps: int


def chunked_mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    rng_key,
    logdensity_fn=None,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    diagonal_preconditioning=True,
    l_factor=0.4,
):
    """:func:`mclmc_find_L_and_step_size` split into resumable chunks.

    Takes the arguments of :func:`mclmc_find_L_and_step_size` except the initial
    state and parameters, which go to ``init``, and returns a
    :class:`ChunkedMCLMCAdaptation`:

    * ``init(state, params=None)`` returns a :class:`MCLMCTuningState` at step 0;
    * ``run_chunk(tuning_state, length)`` runs the next ``length`` tuning steps,
      crossing from one stage to the next when needed;
    * ``final(tuning_state)`` returns ``(state, params, num_tuning_steps)`` as
      :func:`mclmc_find_L_and_step_size` does;
    * ``num_steps`` is the number of tuning steps over the stages: the
      step-size and posterior-size stage, the step-size readjustment after a
      diagonal preconditioning, and the L stage.

    The random keys are those of :func:`mclmc_find_L_and_step_size`, sliced at
    the current step, so chunks covering ``num_steps`` steps give the same
    result as the one-call tuning. ``run_chunk`` is driven from Python: it reads
    ``tuning_state.step`` and calls one compiled scan per stage and chunk length,
    so it must not itself be jitted. The tuning state is a pytree of arrays and
    can be saved between chunks to resume an interrupted tuning; it holds the
    L-stage positions, ``round(num_steps * frac_tune3)`` rows of the dimension.
    """
    if logdensity_fn is None:
        raise ValueError(
            "logdensity_fn is required. Pass the log-density function of the "
            "target distribution."
        )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
        num_steps * frac_tune2
    )
    num_steps3 = round(num_steps * frac_tune3)
    # stage 1: step size and posterior size (make_L_step_size_adaptation)
    tune_keys = jax.random.split(part1_key, num_steps1 + num_steps2 + 1)
    tune_keys, final_key = tune_keys[:-1], tune_keys[-1]
    tune_mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))
    # stage 2: step-size readjustment after the diagonal preconditioning
    readjust_steps = (
        round(num_steps2 / 3) if diagonal_preconditioning and num_steps2 > 1 else 0
    )
    readjust_keys = jax.random.split(final_key, readjust_steps)
    # stage 3: L from the autocorrelation (make_adaptation_L)
    l_steps = num_steps3 if num_steps3 >= 2 else 0
    l_keys = jax.random.split(part2_key, num_steps3)
    stages = (
        ("tune", num_steps1 + num_steps2),
        ("readjust", readjust_steps),
        ("L", l_steps),
    )
    total_steps = sum(length for _, length in stages)
    # as counted by mclmc_find_L_and_step_size
    num_tuning_integrator_steps = (
        num_steps1
        + num_steps2
        + diagonal_preconditioning * (num_steps2 // 3)
        + (num_steps3 if num_steps3 >= 2 else 0)
    )

    def initial_averages(dim):
        return (0.0, 0.0, jnp.inf), (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)]))

    def init(state, params=None):
        flat_position = ravel_pytree(state.position)[0]
        dim = flat_position.shape[0]
        if params is None:
            params = MCLMCAdaptationState(
                jnp.sqrt(dim), jnp.sqrt(dim) * 0.25, inverse_mass_matrix=jnp.ones((dim,))
            )
        adaptive_state, streaming_avg = initial_averages(dim)
        return MCLMCTuningState(
            jnp.asarray(0, dtype=jnp.int32),
            state,
            params,
            jax.tree.map(jnp.asarray, adaptive_state),
            jax.tree.map(jnp.asarray, streaming_avg),
            jnp.zeros((l_steps, dim), dtype=flat_position.dtype),
        )

    compiled = {}

    def stage_scan(name, dim, length):
        # one compiled scan per stage and chunk length
        if (name, length) in compiled:
            return compiled[(name, length)]
        if name == "L":

            def l_step(state, step_input):
                params, key = step_input
                next_state, _ = mclmc_kernel(
                    rng_key=key,
                    state=state,
                    logdensity_fn=logdensity_fn,
                    inverse_mass_matrix=params.inverse_mass_matrix,
                    L=params.L,
                    step_size=params.step_size,
                )
                return next_state, ravel_pytree(next_state.position)[0]

            def run(tuning_state, offset):
                keys = jax.lax.dynamic_slice_in_dim(l_keys, offset, length)
                params = tuning_state.params
                state, positions = jax.lax.scan(
                    lambda s, k: l_step(s, (params, k)), tuning_state.state, keys
                )
                samples = jax.lax.dynamic_update_slice_in_dim(
                    tuning_state.samples, positions, offset, axis=0
                )
                return tuning_state._replace(state=state, samples=samples)

        else:
            step = _make_tuning_step(
                mclmc_kernel,
                logdensity_fn,
                dim,
                desired_energy_var,
                trust_in_estimate,
                num_effective_samples,
            )
            keys_all, mask_all = (
                (tune_keys, tune_mask)
                if name == "tune"
                else (readjust_keys, jnp.ones(readjust_steps))
            )

            def run(tuning_state, offset):
                xs = (
                    jnp.arange(length),
                    jax.lax.dynamic_slice_in_dim(mask_all, offset, length),
                    jax.lax.dynamic_slice_in_dim(keys_all, offset, length),
                )
                carry = (
                    tuning_state.state,
                    tuning_state.params,
                    tuning_state.adaptive_state,
                    tuning_state.streaming_avg,
                )
                (state, params, adaptive_state, streaming_avg), _ = jax.lax.scan(
                    step, carry, xs
                )
                return tuning_state._replace(
                    state=state,
                    params=params,
                    adaptive_state=adaptive_state,
                    streaming_avg=streaming_avg,
                )

        compiled[(name, length)] = jax.jit(run)
        return compiled[(name, length)]

    def end_of_stage(name, tuning_state, dim):
        # what mclmc_find_L_and_step_size does between its scans
        params = tuning_state.params
        if name == "tune":
            if num_steps2 > 1:
                average = tuning_state.streaming_avg[1]
                variances = average[1] - jnp.square(average[0])
                if diagonal_preconditioning:
                    # the readjustment runs with the new mass matrix and the old L
                    params = params._replace(inverse_mass_matrix=variances)
                    adaptive_state, streaming_avg = initial_averages(dim)
                    return tuning_state._replace(
                        params=params,
                        adaptive_state=jax.tree.map(jnp.asarray, adaptive_state),
                        streaming_avg=jax.tree.map(jnp.asarray, streaming_avg),
                    )
                params = params._replace(L=jnp.sqrt(jnp.sum(variances)))
        elif name == "readjust":
            params = params._replace(L=jnp.sqrt(dim))
        else:
            ess = effective_sample_size(tuning_state.samples[None, ...])
            params = params._replace(
                L=l_factor * params.step_size * jnp.mean(l_steps / ess)
            )
        return tuning_state._replace(params=params)

    def run_chunk(tuning_state, length):
        dim = tuning_state.samples.shape[1]
        step = int(tuning_state.step)
        end = min(step + length, total_steps)
        while step < end:
            stage_start = 0
            for name, stage_length in stages:
                if step < stage_start + stage_length:
                    break
                stage_start += stage_length
            offset = step - stage_start
            n = min(end, stage_start + stage_length) - step
            tuning_state = stage_scan(name, dim, n)(tuning_state, offset)
            step += n
            if step == stage_start + stage_length:
                tuning_state = end_of_stage(name, tuning_state, dim)
            tuning_state = tuning_state._replace(step=jnp.asarray(step, dtype=jnp.int32))
        return tuning_state

    def final(tuning_state):
        if int(tuning_state.step) != total_steps:
            raise ValueError(
                f"The tuning is at step {int(tuning_state.step)} of {total_steps}."
            )
        return tuning_state.state, tuning_state.params, num_tuning_integrator_steps

    return ChunkedMCLMCAdaptation(init, run_chunk, final, total_steps)


def _make_tuning_step(
    kernel,
    logdensity_fn,
    dim,
    desired_energy_var,
    trust_in_estimate,
    num_effective_samples,
):
    """One step of the step-size and posterior-size stage of the MCLMC tuning:
    ``step((state, params, adaptive_state, streaming_avg), (_, mask, rng_key))``."""

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(previous_state, params, adaptive_state, rng_key):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        time, x_average, step_size_max = adaptive_state

        rng_key, nan_key = jax.random.split(rng_key)

        # dynamics
        next_state, info = kernel(
            rng_key=rng_key,
            state=previous_state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=params.inverse_mass_matrix,
            L=params.L,
            step_size=params.step_size,
        )

        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
            nan_key,
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight.
        #
        # The step-size adaptation exploits the scaling relation Var[E] = O(eps^6)
        # for the leapfrog integrator (see Bou-Rabee & Sanz-Serna, 2018).
        # xi measures the energy-variance ratio relative to the target; the
        # exponent 6.0 throughout this block originates from that relation.
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # small offset to prevent log(0) divergence
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # Gaussian weight that down-weights step sizes far from the optimum

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # invert the Var[E] = O(eps^6) relation to obtain the optimal step size
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success

    def step(iteration_state, weight_and_key):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        _, mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        x = ravel_pytree(state.position)[0]
        # update the running average of x, x^2
        streaming_avg = incremental_value_update(
            expectation=jnp.array([x, jnp.square(x)]),
            incremental_val=streaming_avg,
            weight=mask * success * params.step_size,
        )

        return (state, params, adaptive_state, streaming_avg), None

    return step


def make_L_step_size_adaptation(
    kernel,
    logdensity_fn,
    dim,
    frac_tune1,
    frac_tune2,
    diagonal_preconditioning,
    desired_energy_var=1e-3,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    progress_bar=False,
    print_rate=None,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for unadjusted MCLMC"""

    step = _make_tuning_step(
        kernel,
        logdensity_fn,
        dim,
        desired_energy_var,
        trust_in_estimate,
        num_effective_samples,
    )

    # NEW: Redefine run_steps to take `length` and use `gen_scan_fn`
    def run_steps(xs, state, params, length):
        scan_fn = gen_scan_fn(length, progress_bar, print_rate)
        mask, keys = xs
        return scan_fn(
            step,
            (
                state,
                params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            (jnp.arange(length), mask, keys),
        )[0]

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
            num_steps * frac_tune2
        )

        L_step_size_adaptation_keys = jax.random.split(
            rng_key, num_steps1 + num_steps2 + 1
        )
        L_step_size_adaptation_keys, final_key = (
            L_step_size_adaptation_keys[:-1],
            L_step_size_adaptation_keys[-1],
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        # NEW: Pass the length for the first scan
        state, params, _, (_, average) = run_steps(
            xs=(mask, L_step_size_adaptation_keys),
            state=state,
            params=params,
            length=num_steps1 + num_steps2,
        )

        L = params.L
        # determine L
        inverse_mass_matrix = params.inverse_mass_matrix
        if num_steps2 > 1:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            L = jnp.sqrt(jnp.sum(variances))

            if diagonal_preconditioning:
                inverse_mass_matrix = variances
                params = params._replace(inverse_mass_matrix=inverse_mass_matrix)
                L = jnp.sqrt(dim)

                # readjust the stepsize
                steps = round(num_steps2 / 3)  # we do some small number of steps
                keys = jax.random.split(final_key, steps)
                state, params, _, (_, average) = run_steps(
                    xs=(jnp.ones(steps), keys), state=state, params=params, length=steps
                )

        return state, MCLMCAdaptationState(L, params.step_size, inverse_mass_matrix)

    return L_step_size_adaptation


def make_adaptation_L(
    kernel, logdensity_fn, frac, l_factor, progress_bar=False, print_rate=None
):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps_3 = round(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps_3)

        def step(state, step_input):
            _, key = step_input
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                logdensity_fn=logdensity_fn,
                inverse_mass_matrix=params.inverse_mass_matrix,
                L=params.L,
                step_size=params.step_size,
            )

            return next_state, next_state.position

        # NEW: Replace standard jax.lax.scan with gen_scan_fn
        scan_fn = gen_scan_fn(num_steps_3, progress_bar, print_rate)
        state, samples = scan_fn(
            step,
            state,
            (jnp.arange(num_steps_3), adaptation_L_keys),
        )

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        ess = effective_sample_size(flat_samples[None, ...])

        return state, params._replace(
            L=l_factor * params.step_size * jnp.mean(num_steps_3 / ess)
        )

    return adaptation_L


def handle_nans(
    previous_state, next_state, step_size, step_size_max, kinetic_change, key
):
    """if there are nans, let's reduce the stepsize, and not update the state. The
    function returns the old state in this case."""

    reduced_step_size = 0.8  # multiplicative shrinkage factor applied on NaN recovery
    p, unravel_fn = ravel_pytree(next_state.position)
    q, unravel_fn = ravel_pytree(next_state.momentum)
    nonans = jnp.logical_and(jnp.all(jnp.isfinite(p)), jnp.all(jnp.isfinite(q)))
    state, step_size, kinetic_change = jax.tree.map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )

    state = jax.lax.cond(
        jnp.isnan(next_state.logdensity),
        lambda: state._replace(
            momentum=generate_unit_vector(key, previous_state.position)
        ),
        lambda: state,
    )

    return nonans, state, step_size, kinetic_change
