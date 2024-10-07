"""Probabilistic programs.

Numpyro handlers can be a bit clunky to work with, these classes provides some
convenient methods for sampling, log density evaluation and reparameterization.
In general, all methods should work in the single sample case, and require explicit
vectorization otherwise, for example using ``equinox.filter_vmap`` or ``jax.vmap``.
"""

from abc import abstractmethod
from collections.abc import Iterable

import equinox as eqx
from flowjax.wrappers import unwrap
from jaxtyping import Array, PRNGKeyArray
from numpyro import handlers
from numpyro.infer import reparam

from pyronox.numpyro_utils import (
    get_sample_site_names,
    shape_only_trace,
    trace_to_distribution_transforms,
    trace_to_log_prob,
)


def _check_present(names, data):
    for site in names:
        if site not in data:
            raise ValueError(f"Expected {site} to be provided in data.")


class AbstractProgram(eqx.Module):
    """Abstract class representing a (numpyro) probabilistic program.

    Provides convenient distribution-like methods for common use cases, along with
    validation of data.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def sample(self, key: PRNGKeyArray, *args, **kwargs) -> dict[str, Array]:
        """Sample the joint distribution (including determninistic sites).

        Args:
            key: Jax random key.
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        seeded_model = handlers.seed(unwrap(self), key)
        trace = handlers.trace(seeded_model).get_trace(*args, **kwargs)
        return {
            k: v["value"]
            for k, v in trace.items()
            if v["type"] in ("sample", "deterministic")
        }

    def log_prob(self, data: dict[str, Array], *args, **kwargs):
        """The joint probability under the model.

        Args:
            data: Dictionary of samples, including all random sites in the program
                (deterministic nodes can be provided, but are not required).
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        """"""
        self = unwrap(self)
        self.validate_data(data, *args, **kwargs)
        _check_present(self.site_names(*args, **kwargs), data)
        sub_model = handlers.substitute(self, data)
        trace = handlers.trace(sub_model).get_trace(*args, **kwargs)
        return trace_to_log_prob(trace, reduce=True)

    def sample_and_log_prob(self, key: PRNGKeyArray, *args, **kwargs):
        """Sample and return its log probability.

        In some instances, this will be more efficient than calling each methods
        seperately. To draw multiple samples, vectorize (jax.vmap) over a set of keys.

        Args:
            key: Jax random key.
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        self = unwrap(self)
        trace = handlers.trace(fn=handlers.seed(self, key)).get_trace(*args, **kwargs)
        samples = {
            k: v["value"]
            for k, v in trace.items()
            if v["type"] in ("sample", "deterministic")
        }
        return samples, trace_to_log_prob(trace)

    def validate_data(
        self,
        data: dict[str, Array],
        *args,
        **kwargs,
    ):
        """Validate the data names and shapes are compatible with the model.

        Any subset of data and deterministic sites can be provided for checking.
        Specifically, this validates that the shapes match what is produced by the
        model when ran with *args and **kwargs. Note, if you have batch dimensions in
        data, this function must be vectorized, e.g. using eqx.filter_vmap.

        Args:
            data: The data.
            model: The model.
            *args: Args passed to model when tracing to infer shapes.
            **kwargs: kwargs passed to model when tracing to infer shapes.
        """
        trace = shape_only_trace(self, *args, **kwargs)
        for name, samples in data.items():
            if name not in trace:
                raise ValueError(f"Got {name} which does not exist in trace.")

            trace_shape = trace[name]["value"].shape
            if trace[name]["type"] in ("sample", "deterministic"):
                if trace_shape != data[name].shape:
                    raise ValueError(
                        f"{name} had shape {trace_shape} in the trace, but shape "
                        f"{samples.shape} in data.",
                    )

    def site_names(self, *args, **kwargs) -> set:
        return get_sample_site_names(unwrap(self), *args, **kwargs).all


class ReparameterizedProgram(AbstractProgram):
    """Reparameterize the model using numpyros transform reparam.

    Args:
        program: The program to reparameterize.
        reparam_names: The latent sites to reparameterize.
        reparam: Whether reparameterization is applied.
    """

    program: AbstractProgram
    reparam_names: frozenset[str]

    def __init__(
        self,
        program: AbstractProgram,
        reparam_names: Iterable[str],
    ):
        self.program = program
        self.reparam_names = frozenset(reparam_names)

    def __call__(self, *args, **kwargs):
        """Program, applying reparameterizations if ``self.reparameterized``."""
        self = unwrap(self)
        config = {name: reparam.TransformReparam() for name in self.reparam_names}
        with handlers.reparam(config=config):
            self.program(*args, **kwargs)

    def get_reparam_transforms(self, latents, *args, **kwargs):
        """Infer the deterministic transforms applied by the reparameterization.

        Args:
            latents: data from the data space (not the base space).
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        model = unwrap(self).set_reparam(set_val=False)
        self.validate_data(latents)
        model = handlers.substitute(model, latents)
        model_trace = handlers.trace(model).get_trace(*args, **kwargs)
        transforms = trace_to_distribution_transforms(model_trace)
        return {k: t for k, t in transforms.items() if k in self.reparam_names}

    def latents_to_original_space(
        self,
        latents: dict[str, Array],
        *args,
        **kwargs,
    ) -> dict[str, Array]:
        """Convert latents from the reparameterized space to original space.

        Args:
            latents: The set of latents from the reparameterized space.
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        self = unwrap(self)
        latents = {k: v for k, v in latents.items()}  # Avoid mutating
        self.validate_data(latents)
        model = handlers.condition(self, latents)
        trace = handlers.trace(model).get_trace(*args, **kwargs)

        for name in self.reparam_names:
            latents.pop(f"{name}_base")
            latents[name] = trace[name]["value"]
        return latents
