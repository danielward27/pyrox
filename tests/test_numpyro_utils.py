import re

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from flowjax.distributions import Normal
from flowjax.experimental.numpyro import sample
from numpyro import handlers

from pyrox.numpyro_utils import (
    _ensure_no_downstream_sites,
    sample_site_names,
    trace_to_distribution_transforms,
    trace_to_log_prob,
)


def model(obs=None):
    with numpyro.plate("plate", 5):
        x = sample("x", numpyro.distributions.Normal())
    sample("y", numpyro.distributions.Normal(x), obs=obs)


def test_trace_to_log_prob():
    obs = jnp.arange(5)
    trace = handlers.trace(handlers.seed(model, jr.key(0))).get_trace(obs=obs)
    log_probs = trace_to_log_prob(trace, reduce=False)

    assert log_probs["x"].shape == (5,)
    assert log_probs["y"].shape == (5,)
    assert pytest.approx(trace["y"]["value"]) == obs
    assert pytest.approx(log_probs["x"]) == Normal().log_prob(trace["x"]["value"])
    assert pytest.approx(log_probs["y"]) == Normal().log_prob(obs - trace["x"]["value"])


def test_trace_to_log_prob_factor():

    def factor_model(obs=None):
        sample("x", numpyro.distributions.Uniform())
        numpyro.factor("factor", 20)

    trace = handlers.trace(handlers.seed(factor_model, jr.key(0))).get_trace()
    assert pytest.approx(20) == trace_to_log_prob(trace, reduce=True)


def test_trace_to_distribution_transforms():

    def model(obs=None):
        with numpyro.plate("plate", 5):
            x = sample("x", Normal())

        sample("y", Normal(x), obs=obs)

    data = {"x": jnp.arange(5)}
    trace = handlers.trace(handlers.condition(model, data)).get_trace(obs=jnp.zeros(5))
    transforms = trace_to_distribution_transforms(trace)

    assert pytest.approx(transforms["x"](jnp.zeros(5))) == jnp.zeros(5)
    assert pytest.approx(transforms["y"](jnp.zeros(5))) == data["x"]

    def nested_plate_model():
        with numpyro.plate("plate1", 2):
            with numpyro.plate("plate1", 3):
                sample("x", Normal(0, 2))

    trace = handlers.trace(handlers.condition(nested_plate_model, {"x": 1})).get_trace()
    transforms = trace_to_distribution_transforms(trace)
    assert transforms["x"](1) == 2


def test_get_sample_site_names():
    names = sample_site_names(model)
    assert names.observed == set()
    assert names.latent == {"x", "y"}
    assert names.all == {"x", "y"}

    names = sample_site_names(model, obs=jnp.array(0))
    assert names.observed == {"y"}
    assert names.latent == {"x"}
    assert names.all == {"x", "y"}


test_cases = (
    (["a"], "Site b can not be downstream of a."),
    (["a", "b"], "Site c can not be downstream of b"),
    (["b"], "Site c can not be downstream of b."),
)


@pytest.mark.parametrize(("sites", "match"), test_cases)
def test_ensure_no_downstream_sites(sites, match):

    def model_1():
        a = numpyro.sample("a", numpyro.distributions.Normal(0, 1))
        b = numpyro.sample("b", numpyro.distributions.Normal(a, 1), obs=1)
        numpyro.sample("c", numpyro.distributions.Normal(b, 1))

    with pytest.raises(ValueError, match=re.escape(match)):
        _ensure_no_downstream_sites(model_1, sites)
