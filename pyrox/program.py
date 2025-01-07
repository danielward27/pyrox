"""Probabilistic programs.

Numpyro handlers can be a bit clunky to work with, these classes provides some
convenient methods for sampling, log density evaluation and reparameterization.
In general, all methods should work in the single sample case, and require explicit
vectorization otherwise, for example using ``equinox.filter_vmap`` or ``jax.vmap``.
"""

from abc import abstractmethod
from collections.abc import Iterable

import equinox as eqx
import jax
from jax import ShapeDtypeStruct
from jaxtyping import Array, PRNGKeyArray, PyTree
from numpyro import handlers
from numpyro.distributions import transforms as ntransforms
from numpyro.infer import reparam
from paramax import wrappers
from paramax.wrappers import unwrap

from pyrox.numpyro_utils import (
    sample_site_names,
    shape_only_trace,
    trace_to_distribution_transforms,
    trace_to_log_prob,
)


def _check_present(names, data):
    for site in names:
        if site not in data:
            raise ValueError(f"Expected {site} to be provided in data.")


class _DistributionLike(eqx.Module):
    # Shared between AbstractProgram and GuideToDataSpace

    @abstractmethod
    def sample(self, key: PRNGKeyArray, **kwargs) -> dict[str, Array]:
        pass

    @abstractmethod
    def log_prob(self, data: dict[str, Array], **kwargs):
        pass

    @abstractmethod
    def sample_and_log_prob(self, key: PRNGKeyArray, **kwargs):
        pass


class AbstractProgram(_DistributionLike):
    """Abstract class representing a (numpyro) probabilistic program.

    Provides convenient distribution-like methods for common use cases, along with
    validation of data.
    """

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    def sample(
        self,
        key: PRNGKeyArray,
        condition: dict | None = None,
        **kwargs,
    ) -> dict[str, Array]:
        """Sample the joint distribution (including determninistic sites).

        Args:
            key: Jax random key.
            condition: Any sites to fill in with handlers.condition
            **kwargs: Key word arguments passed to the program.
        """
        model = handlers.seed(unwrap(self), key)
        if condition is not None:
            model = handlers.condition(model, condition)

        trace = handlers.trace(model).get_trace(**kwargs)
        return {
            k: v["value"]
            for k, v in trace.items()
            if v["type"] in ("sample", "deterministic")
        }

    def log_prob(self, data: dict[str, Array], **kwargs):
        """The joint probability under the model.

        Args:
            data: Dictionary of samples, including all latent sites in the program
                (deterministic nodes can be provided, but are not required).
            **kwargs: Key word arguments passed to the program.
        """
        """"""
        self = unwrap(self)
        self.validate_data(data, **kwargs)
        _check_present(self.site_names(**kwargs).latent, data)
        sub_model = handlers.substitute(self, data)
        trace = handlers.trace(sub_model).get_trace(**kwargs)
        return trace_to_log_prob(trace, reduce=True)

    def sample_and_log_prob(self, key: PRNGKeyArray, **kwargs):
        """Sample and return its log probability.

        In some instances, this will be more efficient than calling each methods
        seperately. To draw multiple samples, vectorize (jax.vmap) over a set of keys.

        Args:
            key: Jax random key.
            **kwargs: Key word arguments passed to the program.
        """
        self = unwrap(self)
        trace = handlers.trace(fn=handlers.seed(self, key)).get_trace(**kwargs)
        samples = {
            k: v["value"]
            for k, v in trace.items()
            if v["type"] in ("sample", "deterministic")
        }
        return samples, trace_to_log_prob(trace)

    def validate_data(
        self,
        data: dict[str, Array],
        **kwargs,
    ):
        """Validate the data names and shapes are compatible with the model.

        Any subset of data and deterministic sites can be provided for checking.
        Specifically, this validates that the shapes match what is produced by the
        model when ran with **kwargs. Note, if you have batch dimensions in
        data, this function must be vectorized, e.g. using eqx.filter_vmap.

        Args:
            data: The data.
            model: The model.
            **kwargs: kwargs passed to model when tracing to infer shapes.
        """
        trace = shape_only_trace(self, **kwargs)
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

    def site_names(self, **kwargs):
        """Returns a named tuple with elements, latent, observed and all."""
        return sample_site_names(unwrap(self), **kwargs)

    def get_prior(self, observed_sites: Iterable[str]):
        """Return the program representing the prior distribution.

        Note this assumes that no latents are downstream of the observed sites.

        Args:
            observed_sites: The observed sites in the model.
        """
        return _ProgramToPrior(self, observed_sites)


class SetKwargs(AbstractProgram):
    """Wraps a program, setting key word arguments for the program.

    We assume the key word arguments are non-trainable.
    """

    program: AbstractProgram
    kwargs: wrappers.NonTrainable | dict

    def __init__(self, program: AbstractProgram, **kwargs):
        self.program = program
        self.kwargs = wrappers.NonTrainable(kwargs)

    def __call__(self, **kwargs):
        return self.program(**kwargs, **wrappers.unwrap(self.kwargs))


class _ProgramToPrior(AbstractProgram):
    program: AbstractProgram
    observed_sites: frozenset

    def __init__(self, program: AbstractProgram, observed_sites: Iterable, **kwargs):
        self.program = program
        self.observed_sites = frozenset(observed_sites)

    def __call__(self, **kwargs):
        # We provide dummy data otherwise numpyro will still generate observed data.
        dummy_data = {k: ShapeDtypeStruct((), float) for k in self.observed_sites}
        model = handlers.condition(self.program, dummy_data)
        return handlers.block(model, hide=self.observed_sites)(**kwargs)


class ReparameterizedProgram(AbstractProgram):
    """Reparameterize the model using numpyros transform reparam.

    Currenlty only supports TransformReparam and ExplicitReparam from numpyro.

    Args:
        program: The program to reparameterize.
        config: The dictionary mapping the string to the reparameterization.
    """

    program: AbstractProgram
    config: dict[str, reparam.TransformReparam | reparam.ExplicitReparam]

    def __init__(
        self,
        program: AbstractProgram,
        config: dict[str, reparam.TransformReparam | reparam.ExplicitReparam],
    ):
        self.program = program
        self.config = config

    def __call__(self, **kwargs):
        """Program applying reparameterizations."""
        self = unwrap(self)
        with handlers.reparam(config=self.config):
            self.program(**kwargs)

    def latents_to_original_space(
        self,
        latents: dict[str, Array],
        **kwargs,
    ) -> dict[str, Array]:
        """Convert latents from the reparameterized space to original space.

        Args:
            latents: The set of latents from the reparameterized space.
            **kwargs: Key word arguments passed to the program.
        """
        self = unwrap(self)
        _check_present(self.site_names(**kwargs).latent, latents)
        latents = {k: v for k, v in latents.items()}  # Avoid mutating
        self.validate_data(latents, **kwargs)
        model = handlers.condition(self, latents)
        trace = handlers.trace(model).get_trace(**kwargs)

        for name in self.config.keys():
            latents.pop(f"{name}_base")
            latents[name] = trace[name]["value"]
        return latents

    def _infer_reparam_transforms(
        self,
        latents,
        **kwargs,
    ) -> dict[str, ntransforms.Transform]:  # TODO is this tested?
        """Infer transforms used for reparameterizing.

        Only supports ExplicitReparam and TransformReparam sites.

        Args:
            latents: data from the data space (not the base space).
            **kwargs: Key word arguments passed to the program.
        """
        self.validate_data(latents, **kwargs)
        _check_present(self.program.site_names(**kwargs).latent, latents)
        program = handlers.substitute(self.program, latents)
        program_trace = handlers.trace(program).get_trace(**kwargs)
        trace_transforms = trace_to_distribution_transforms(program_trace)

        transforms = {}
        for k, repar in self.config.items():
            if isinstance(repar, reparam.TransformReparam):
                transforms[k] = trace_transforms[k].inv

            elif isinstance(repar, reparam.ExplicitReparam):
                transforms[k] = repar.transform

        return transforms


class GuideToDataSpace(_DistributionLike):
    """Guide in data/original space, by inverting model reparameterizations.

    Often after fitting, we want to convert the guide back to the original
    space of the program without reparameterizations. This class achieves this.
    Make sure you pass the observations, so
    """

    guide: AbstractProgram
    model: ReparameterizedProgram
    guide_kwargs: dict
    model_kwargs: dict

    def __init__(
        self,
        *,
        guide: AbstractProgram,
        model: ReparameterizedProgram,
        guide_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ):
        self.guide = guide
        self.model = model
        self.guide_kwargs = {} if guide_kwargs is None else guide_kwargs
        self.model_kwargs = {} if model_kwargs is None else model_kwargs

    def sample(
        self,
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        latents = self.guide.sample(key, **self.guide_kwargs)
        latents = {
            k: latents[k] for k in self.model.site_names(**self.model_kwargs).latent
        }
        return self.model.latents_to_original_space(latents, **self.model_kwargs)

    def log_prob(
        self,
        data: dict[str, Array],
    ):
        """Compute guide probability of latents in original space.

        We achieve this by reparameterizing the guide with the inverse of the model
        reparameterization transforms.
        """
        reparam_transforms = self.model._infer_reparam_transforms(
            data,
            **self.model_kwargs,
        )
        reparam_transforms = {
            f"{k}_base": reparam.ExplicitReparam(t.inv)
            for k, t in reparam_transforms.items()
        }
        guide = ReparameterizedProgram(self.guide, reparam_transforms)
        # Numpyro doesn't support choosing the name modification
        # We have to match _base_base names introduced by guide reparameterization.
        data = {
            f"{k}_base_base" if k in self.model.config else k: v
            for k, v in data.items()
        }
        return guide.log_prob(data, **self.guide_kwargs)

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
    ):
        # Assuming efficiency isn't vital here.
        sample = self.sample(key)
        log_prob = self.log_prob(sample)
        return sample, log_prob


def remove_reparam(tree: PyTree):
    """Remove reparameterizations from a pytree.

    This maps over the pytree, and unwraps the ``ReparameterizedProgram``s.
    """

    def replace_fn(leaf):
        if isinstance(leaf, ReparameterizedProgram):
            return leaf.program
        return leaf

    return jax.tree.map(
        replace_fn,
        tree,
        is_leaf=lambda leaf: isinstance(leaf, ReparameterizedProgram),
    )
