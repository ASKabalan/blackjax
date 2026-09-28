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
"""Chunked window and MCLMC adaptations reproduce the one-call adaptations,
including when resumed from a state saved between chunks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax
from blackjax.mcmc.integrators import isokinetic_mclachlan

COV = jnp.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 0.01]])


def logdensity_fn(x):
    return -0.5 * x @ jnp.linalg.solve(COV, x)


def save_and_load(tree):
    # a checkpoint: device arrays -> host arrays -> device arrays
    return jax.tree.map(jnp.asarray, jax.tree.map(np.asarray, tree))


def assert_trees_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_chunked_window_adaptation_matches_run():
    num_steps, chunk = 100, 7
    rng_key = jax.random.key(0)
    position = jnp.ones(3)
    kwargs = dict(algorithm=blackjax.nuts, logdensity_fn=logdensity_fn)
    expected, _ = blackjax.window_adaptation(**kwargs).run(rng_key, position, num_steps)

    chunked = blackjax.chunked_window_adaptation(**kwargs)
    run_chunk = jax.jit(chunked.run_chunk, static_argnums=(2, 3))
    warmup_state = chunked.init(position)
    done = 0
    while done < num_steps:
        length = min(chunk, num_steps - done)
        warmup_state, _ = run_chunk(warmup_state, rng_key, num_steps, length)
        done += length
        if done == 2 * chunk:
            warmup_state = save_and_load(warmup_state)
    assert int(warmup_state.step) == num_steps
    result = chunked.final(warmup_state)

    assert_trees_equal(result.state, expected.state)
    assert_trees_equal(result.parameters, expected.parameters)


@pytest.mark.parametrize("diagonal_preconditioning", [False, True])
def test_chunked_mclmc_adaptation_matches_one_call(diagonal_preconditioning):
    num_steps, chunk = 200, 9
    rng_key, init_key = jax.random.split(jax.random.key(1))
    kernel = blackjax.mcmc.mclmc.build_kernel(integrator=isokinetic_mclachlan)
    initial_state = blackjax.mcmc.mclmc.init(
        position=jnp.ones(3), logdensity_fn=logdensity_fn, rng_key=init_key
    )
    kwargs = dict(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        rng_key=rng_key,
        logdensity_fn=logdensity_fn,
        frac_tune1=0.4,
        frac_tune2=0.4,
        frac_tune3=0.2,
        diagonal_preconditioning=diagonal_preconditioning,
    )
    expected_state, expected_params, expected_steps = (
        blackjax.mclmc_find_L_and_step_size(state=initial_state, **kwargs)
    )

    chunked = blackjax.chunked_mclmc_find_L_and_step_size(**kwargs)
    tuning_state = chunked.init(initial_state)
    while int(tuning_state.step) < chunked.num_steps:
        tuning_state = chunked.run_chunk(tuning_state, chunk)
        if int(tuning_state.step) == 2 * chunk:
            tuning_state = save_and_load(tuning_state)
    state, params, steps = chunked.final(tuning_state)

    assert steps == expected_steps
    assert_trees_equal(state, expected_state)
    assert_trees_equal(params, expected_params)
