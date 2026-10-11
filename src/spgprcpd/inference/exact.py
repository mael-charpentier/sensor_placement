"""Exact Gaussian-process regression, in three variants.

All three condition on the full training set (no inducing points, no approximation) and
differ in exactly one thing: how often the kernel hyperparameters are fitted.  That makes
them a ladder, each rung a further restriction of the one above:

    Exact_GP                    fit on every batch
    └── Exact_GP_no_retraining  fit once, on the first batch, then only grow the data
        └── Exact_GP_no_training  never fit

Why the ladder is worth having:

  * Exact_GP is the realistic online model. It re-estimates the kernel as data arrives and
    pays a full optimisation per batch for it. Note that it restarts from
    initial_learning_kernel each time rather than warm-starting from the previous fit.
  * Exact_GP_no_retraining isolates the cost of re-estimation. Same conditioning set as
    Exact_GP and the same first fit, but the hyperparameters are frozen afterwards, so the
    gap between the two is what continuous re-estimation buys.
  * Exact_GP_no_training is the oracle. It never estimates anything, so the caller hands it
    the true generating kernel and the true observation noise, see experiments/synthetic.py.
    It is the upper reference the learned models are measured against, not a method anyone
    could run in practice.
"""
import copy

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import parameter_dict

from spgprcpd.inference.base import Extrapolation


class Exact_GP(Extrapolation):
    """Exact gaussian process that refits its hyperparameters on every batch (module docstring).

    Parameters:
    kernel: gpflow.kernels.Kernel
        Starting point for the fit.  Every create restarts from this kernel, so it is
        never mutated by the optimiser.
    noise_variance: float
        Observation noise sigma_eps^2.  The fitting variants learn their own. The oracle
        cannot, so it must be given the true value.
    attempts: int
        Number of optimiser restarts. The highest-likelihood fit wins.
    from_scratch: bool
        Rebuild and refit from the initial kernel on every update, discarding the
        accumulated model (the data is still accumulated).
    """

    def __init__(self, kernel, noise_variance=1e-3, attempts=1, from_scratch=False):
        super().__init__(kernel, attempts=attempts)
        self.noise_variance = noise_variance
        self.from_scratch = from_scratch

    # building blocks, the ladder below overrides these, not create
    def _build(self, X, y):
        """Construct the gaussian process regression on (X, y) and set its observation noise."""
        self.model = gpflow.models.GPR(
            data=(X, y),
            kernel=self.initial_learning_kernel,
            mean_function=None,
        )
        self.model.likelihood.variance.assign(self.noise_variance)

    def _fit(self):
        """Fit the hyperparameters.  Exact_GP_no_training overrides this to do nothing."""
        self._optimize()

    def _likelihood(self):
        """Log marginal likelihood recorded in the history.

        A variant that never fits overrides this: reporting a likelihood for a model whose
        hyperparameters were handed to it invites reading it as a fit quality.
        """
        return self.model.log_marginal_likelihood().numpy()

    def _record_state(self):
        if self.model is None:
            self._history['likelihood'].append(np.nan)
            self._history['kernel_params'].append({
                "kernel_name": np.nan,
                "kernel_param": np.nan,
            })
            return
        self._history['likelihood'].append(self._likelihood())
        self._history['kernel_params'].append({
            "kernel_name": str(type(self.model.kernel)),
            "kernel_param": {k: v.numpy()
                             for k, v in parameter_dict(self.model.kernel).items()},
        })

    def create(self, X, y):
        self._build(X, y)
        self._fit()
        self._record_state()

    def _optimize(self, **options):
        best_model = None
        best_likelihood = -np.inf

        def fit():
            model_copy = self.copy_model()
            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(
                model_copy.training_loss,
                variables=model_copy.trainable_variables,
                options=options
            )
            return model_copy, model_copy.log_marginal_likelihood()

        for _ in range(self.attempts):
            model_copy, likelihood = self._optimize_with_jitter(fit)

            if likelihood >= best_likelihood or best_model is None:
                best_likelihood = likelihood
                best_model = model_copy

        self.model = best_model

    def _accumulate(self, X, y):
        """Merge a new batch into the stored training data. Returns (X_all, Y_all).

        Two accumulation modes, told apart by whether the batch repeats the same inputs:

          * the point cloud mode, time as an input feature. Each batch brings new (x, y)
            rows, so both are row-stacked. Consecutive batches have different inputs because
            X carries a per-batch time column, hence old_X is not X.
          * the snapshot mode, also called multi-output, selected by time_output. The same
            locations are observed at new time steps, with y holding one snapshot per
            column, so the snapshot columns are stacked and the inputs left unchanged. This
            is what lets batches of unequal length combine, as in a phase of 50 then 50 then
            30 that does not divide evenly. A row-stack cannot do that, because the
            transposed y blocks would then have mismatched column counts.
        """
        old_X, old_Y = self.model.data
        old_X = np.asarray(old_X, float)
        old_Y = np.asarray(old_Y, float)
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        same_inputs = (old_X.shape == X.shape and old_Y.shape[0] == y.shape[0]
                       and np.allclose(old_X, X))
        if same_inputs:
            return X, np.hstack([old_Y, y])
        return np.vstack([old_X, X]), np.vstack([old_Y, y])

    def update(self, X, y):
        if self.model is None or self.from_scratch:
            self.create(X, y)
        else:
            self.create(*self._accumulate(X, y))

    def predict(self, pred_x):
        """Accumulate predictions into history, flattening column-major (.T then flatten)."""
        mx, Sx = self.predict_f(pred_x)
        mx_ = np.asarray(mx.numpy() if hasattr(mx, "numpy") else mx)
        Sx_ = np.asarray(Sx.numpy() if hasattr(Sx, "numpy") else Sx)
        self._history['means'].extend(mx_.T.flatten())
        self._history['variances'].extend(Sx_.T.flatten())
        return mx, Sx

    def copy(self):
        """A detached copy, sharing no mutable state."""
        new_obj = type(self)(self.initial_learning_kernel,
                             noise_variance=self.noise_variance,
                             attempts=self.attempts,
                             from_scratch=self.from_scratch)
        new_obj._history["likelihood"] = self._history["likelihood"].copy()
        new_obj._history["kernel_params"] = copy.deepcopy(self._history["kernel_params"])
        if self.model is not None:
            new_obj.model = self.copy_model()
        return new_obj


class Exact_GP_no_retraining(Exact_GP):
    """Exact gaussian process fitted once, on the first batch, then frozen (module docstring).

    Later batches only enlarge the conditioning set: the accumulated data is swapped into
    the existing model in place, so no gaussian process regression is rebuilt and no optimiser runs.  Prediction
    still uses the full accumulated data, so this is an exact gaussian process throughout, the only
    thing held fixed is the kernel.

    Comparing it against Exact_GP isolates the value of continuous re-estimation.    comparing it against Exact_GP_no_training isolates the value of estimating the
    kernel at all.
    """

    def update(self, X, y):
        if self.model is None or self.from_scratch:
            super().update(X, y)          # first batch: build and fit through create
            return
        X_all, Y_all = self._accumulate(X, y)
        # Swap the data into the existing model rather than rebuilding it: the kernel and
        # the noise are deliberately left exactly as the first fit left them.
        self.model.data = (
            tf.constant(X_all, dtype=tf.float64),
            tf.constant(Y_all, dtype=tf.float64),
        )
        self._record_state()


class Exact_GP_no_training(Exact_GP_no_retraining):
    """Exact gaussian process that never fits, the true-model oracle (module docstring).

    Because it estimates nothing, the caller must supply both the kernel and the observation
    noise. Experiments/synthetic.py hands it a fresh copy of the true generating kernel
    and data_manager.noise_std ** 2.  Given the wrong kernel it will happily produce
    confident nonsense, that is the point, it has no way to notice.

    It inherits the in-place accumulation of its parent, which for a model that never fits
    is pure saving: rebuilding a gaussian process regression you are not going to optimise changes nothing.
    """

    def _fit(self):
        """No fitting, ever: the hyperparameters are the ones the caller supplied."""

    def _likelihood(self):
        """nan rather than a number.

        The marginal likelihood is computable here, but recording it would put a value in
        the same history slot the fitting variants use for their achieved fit quality, and
        it would be read as one. This model was never fitted, so it reports nothing.
        """
        return np.nan
