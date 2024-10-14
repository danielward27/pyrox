import re

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from flowjax.bijections import Exp, Invert
from flowjax.distributions import LogNormal, Normal
from flowjax.experimental.numpyro import _BijectionToNumpyro, sample
from numpyro.infer import reparam

from pyrox.program import AbstractProgram, GuideToDataSpace, ReparameterizedProgram


class Program(AbstractProgram):

    def __call__(self, *, obs=None):
        scale = sample("scale", LogNormal())
        with numpyro.plate("plate", 5):
            x = sample("x", Normal(1, scale))
        y = numpyro.deterministic("y", x + 1)
        sample("z", Normal(y), obs=obs)


def reparameterized_program():
    return ReparameterizedProgram(
        Program(),
        {
            "x": reparam.TransformReparam(),
            "scale": reparam.ExplicitReparam(_BijectionToNumpyro(Invert(Exp()))),
        },
    )


class Guide(AbstractProgram):

    def __call__(self, obs=None):
        sample("scale_base", Normal())
        with numpyro.plate("plate", 5):
            sample("x_base", Normal())


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
    program.validate_data({"y": jnp.ones(5)})


def test_sample():
    sample = Program().sample(key=jr.key(0))
    assert sample.keys() == {"scale", "x", "y", "z"}


def test_log_prob():
    model = Program()
    sample = model.sample(key=jr.key(0))
    log_prob = model.log_prob(sample)
    assert log_prob.shape == ()

    # Test without deterministic site same result
    sample.pop("y")
    log_prob2 = model.log_prob(sample)
    assert log_prob == log_prob2

    # Test passing obs same result
    obs = sample.pop("z")
    log_prob3 = model.log_prob(sample, obs=obs)
    assert log_prob == log_prob3


def test_sample_and_log_prob():
    program = Program()
    key = jr.key(0)
    sample, log_prob = program.sample_and_log_prob(key)
    assert sample.keys() == {"scale", "x", "y", "z"}

    sample2 = program.sample(key)
    log_prob2 = program.log_prob(sample2)

    assert sample.keys() == sample2.keys()
    assert all(
        pytest.approx(v1) == v2
        for v1, v2 in zip(sample.values(), sample2.values(), strict=True)
    )
    assert pytest.approx(log_prob2) == log_prob


def test_reparameterized_program():
    key = jr.key(0)
    # Original space sample and log prob
    sample_original, log_prob1 = Program().sample_and_log_prob(key)

    program = reparameterized_program()
    base_sample = program.sample(key)
    assert "x_base" in base_sample
    assert "scale_base" in base_sample

    base_sample_lp = program.sample_and_log_prob(key)
    assert "x_base" in base_sample_lp[0]
    assert "scale_base" in base_sample_lp[0]

    # Map back to data space
    reconstructed = program.latents_to_original_space(base_sample)
    assert sample_original.keys() == reconstructed.keys()
    assert all(
        pytest.approx(sample_original[k]) == reconstructed[k]
        for k in reconstructed.keys()
    )

    # Test _infer_reparam_transforms
    reparam = program._infer_reparam_transforms(base_sample)
    assert {"scale", "x"} == reparam.keys()

    x = 2
    assert pytest.approx(jnp.log(x)) == reparam["scale"](x)

    expected = (x - 1) / sample_original["scale"]
    assert pytest.approx(expected) == reparam["x"](x)


def test_guide_to_data_space():

    key = jr.key(0)
    guide = GuideToDataSpace(Guide(), reparameterized_program())
    sample = guide.sample(key, obs=jnp.arange(5))
    assert "scale" in sample
    assert "x" in sample

    assert guide.log_prob(sample, obs=jnp.arange(5)).shape == ()
    # TODO lazy test above.


def test_prior():
    model = Program()
    prior = model.get_prior(observed_sites=["z"])
    samp = prior.sample(jr.key(0))
    log_prob = prior.log_prob(samp)

    expected = sum(
        [
            LogNormal().log_prob(samp["scale"]).sum(),
            Normal(1, samp["scale"]).log_prob(samp["x"]).sum(),
        ],
    )

    assert pytest.approx(expected) == log_prob
