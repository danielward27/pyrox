# pyrox

Provides a more pytorch-like interface for numpyro, using
[equinox](https://github.com/patrick-kidger/equinox) and
[flowjax](https://github.com/danielward27/flowjax).

- Includes distribution-like methods for probabilistic programs (``log_prob``,
``sample``, etc.).
- Simplifies reparameterization handling for variantional inference.
- Includes [SoftCVI](https://arxiv.org/abs/2407.15687).

Although it makes some things much simpler, it also make some things clunky too - this package should be considered experimental.
