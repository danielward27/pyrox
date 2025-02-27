"""Numpyro utility functions."""

from collections import namedtuple
from collections.abc import Callable, Iterable
from functools import partial

import jax.random as jr
from jax import eval_shape
from jax.tree_util import Partial, tree_map
from numpyro import distributions as ndist
from numpyro import handlers
from numpyro.distributions import transforms as ntransforms
from numpyro.distributions.util import is_identically_one
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.inspect import get_model_relations
from numpyro.ops.pytree import PytreeTrace


def shape_only_trace(model: Callable, **kwargs):
    """Trace the numpyro model using ``jax.eval_shape``, avoiding array flops.

    Note callables are wrapped to  are also removed from the output (not)

    Args:
       model: The numpyro model.
       args: Arguments passed to model.
       kwargs: Key word arguments passed to model.
    """

    # Adapted from https://github.com/pyro-ppl/numpyro/blob/5af9ebda72bd7aeb08c61e4248ecd0d982473224/numpyro/infer/inspect.py#L39
    def get_trace(fn):
        fn = handlers.substitute(
            handlers.seed(model, jr.key(0)),
            substitute_fn=init_to_sample,
        )  # Support improper uniform
        fn = handlers.seed(fn, jr.key(0))
        trace = handlers.trace(fn).get_trace(**kwargs)

        # We wrap all callables to ensure return value are all valid jax types
        trace = tree_map(
            lambda leaf: Partial(leaf) if callable(leaf) else leaf,
            tree=trace,
            is_leaf=callable,
        )
        return PytreeTrace(trace)

    trace = eval_shape(lambda: get_trace(model)).trace

    # unwrap the result
    return tree_map(
        lambda leaf: leaf.func if isinstance(leaf, Partial) else leaf,
        tree=trace,
        is_leaf=lambda leaf: isinstance(leaf, Partial),
    )


def sample_site_names(model: Callable, **kwargs):
    """Infer the names of the sample sites of a model given args and kwargs.

    Args:
        model: Model from which to infer the latents.
        **kwargs: Key word arguments passed to the model.

    Returns:
        A dataclass with ``observed``, ``latent`` and ``all`` field/property names.
    """
    trace = shape_only_trace(model, **kwargs)
    observed, latent = set(), set()
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        if site["is_observed"]:
            observed.add(name)
        else:
            latent.add(name)

    SiteNames = namedtuple("SiteNames", ["observed", "latent", "all"])

    return SiteNames(
        observed=set(observed),
        latent=set(latent),
        all=set(observed) | set(latent),
    )


def trace_to_distribution_transforms(trace: dict):
    """Get the numpyro transforms any transformed distributions in trace.

    Note if TransformReparam is used for a distribution, then the transform will not be
    extracted from the trace as the transform is instead treated as a deterministic
    function.
    """
    transforms = {}
    for k, site in trace.items():

        if site["type"] == "sample":
            dist = site["fn"]

            while isinstance(dist, ndist.ExpandedDistribution | ndist.Independent):
                dist = dist.base_dist

            if isinstance(dist, ndist.TransformedDistribution):
                if len(dist.transforms) == 1:
                    transforms[k] = dist.transforms[0]
                else:
                    transforms[k] = ntransforms.ComposeTransform(dist.transforms)

    return transforms


def eval_site_log_prob(site: dict):
    """Evaluate the log probability of a site."""
    log_prob_fn = site["fn"].log_prob

    if site["intermediates"]:
        log_prob_fn = partial(log_prob_fn, intermediates=site["intermediates"])

    log_prob = log_prob_fn(site["value"])

    if site["scale"] is not None and not is_identically_one(site["scale"]):
        log_prob = site["scale"] * log_prob

    return log_prob


def trace_to_log_prob(trace: dict, *, reduce: bool = True):
    """Computes dictionary of log probabilities, or a scalar value for a trace.

    Args:
        trace: The numpyro trace.
        reduce: Whether to reduce the result to a scalar value.
    """
    log_prob = {
        k: eval_site_log_prob(site)
        for k, site in trace.items()
        if site["type"] == "sample"
    }
    if reduce:
        log_prob = sum(v.sum() for v in log_prob.values())
    return log_prob


def _ensure_no_downstream_sites(model, sites: Iterable[str], **kwargs):
    """Check no sites are dowmstream of the set a set of sites."""
    relations = get_model_relations(model, model_kwargs=kwargs)["sample_sample"]
    relations = {k: v for k, v in relations.items() if k not in sites}

    for latent_name, rel in relations.items():
        for r_site in rel:
            if r_site in sites:
                raise ValueError(
                    f"Site {latent_name} can not be downstream of {r_site}.",
                )

