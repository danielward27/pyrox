import re

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from flowjax.distributions import Normal
from flowjax.experimental.numpyro import sample

from pyronox.program import AbstractProgram, ReparameterizedProgram


class Program(AbstractProgram):

    def __call__(self, obs=None):
        with numpyro.plate("plate", 5):
            x = sample("x", Normal())
        y = numpyro.deterministic("y", x + 1)
        sample("z", Normal(y), obs=obs)


validate_data_raises_test_cases = {
    "invalid_site_name": (
        {"q": jnp.ones(())},
        "q which does not exist in trace",
    ),
    "invalid_sample_shape": (
        {"x": jnp.ones((2,))},
        "x had shape (5,) in the trace, but shape (2,) in data.",
    ),
    "invalid_deterministic_shape": (
        {"y": jnp.ones((2, 2))},
        "y had shape (5,) in the trace, but shape (2, 2) in data.",
    ),
}


@pytest.mark.parametrize(
    ("data", "match"),
    validate_data_raises_test_cases.values(),
    ids=validate_data_raises_test_cases.keys(),
)
def test_validate_data_raises(data, match):
    program = Program()
    with pytest.raises(ValueError, match=re.escape(match)):
        program.validate_data(data)


def test_validate_data_does_not_raise():
    program = Program()
    data = {"x": jnp.ones(5), "y": jnp.ones(5), "z": jnp.ones(5)}
    assert program.validate_data(data) is None

    # Subset of data
    program.validate_data({"y": jnp.ones(5)}, data)


def test_sample():
    sample = Program().sample(key=jr.key(0))
    assert sample.keys() == {"x", "y", "z"}


def test_log_prob():
    model = Program()
    sample = model.sample(key=jr.key(0))
    log_prob = model.log_prob(sample)
    assert log_prob.shape == ()

    # Test without deterministic site same result
    sample.pop("y")
    log_prob2 = model.log_prob(sample)
    assert log_prob == log_prob2


def test_sample_and_log_prob():
    program = Program()
    sample, log_prob = program.sample_and_log_prob(jr.key(0))
    assert sample.keys() == {"x", "y", "z"}

    log_prob2 = program.log_prob(sample)
    assert pytest.approx(log_prob2) == log_prob


def test_reparameterized_program():
    key = jr.key(0)
    # Original space sample and log prob
    sample_original, log_prob1 = Program().sample_and_log_prob(key)

    program = ReparameterizedProgram(Program(), {"x"})
    sample = program.sample(key)
    assert "x_base" in sample

    sample, _ = program.sample_and_log_prob(key)
    assert "x_base" in sample

    # Map back to x space
    sample = program.latents_to_original_space(sample)
    assert sample_original.keys() == sample.keys()
    assert all(pytest.approx(sample_original[k]) == sample[k] for k in sample.keys())
