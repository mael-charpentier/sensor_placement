"""Streaming sparse gaussian process: an online model summarising past data by inducing points.

Unlike the exact ladder in spgprcpd.inference.exact, this does not keep its
training data, each batch is absorbed into a fixed-size set of inducing points. That is
what makes it usable on a long stream, and also why it cannot serve as the f0 estimation
model (reading a posterior mean off it needs the training set it discarded).

derived work. Adapted from https://github.com/thangbui/streaming_sparse_gp
(Bui, Nguyen & Turner, 2017). Upstream licence: Apache-2.0

modified from the original, as Apache-2.0 section 4(b) requires this notice to state:
wrapped in this project's spgprcpd.inference.base.Extrapolation interface, with
the jitter-retry fitting and the prediction-history accumulation this codebase needs.
"""
import copy

import gpflow
import numpy as np
from gpflow.utilities import parameter_dict

from spgprcpd.inference.base import Extrapolation
from spgprcpd.inference.pep import OSGPR_PEP, SGPR_PEP


class Streaming_Sparse_GP(Extrapolation):
    def __init__(self, kernel, num_induce_points=40, *,
                 use_old_Z = False, alpha = 0.5):
        super().__init__(kernel)
        self.M = num_induce_points

        self.use_old_Z = use_old_Z
        self.alpha = alpha

    def init_Z(self, cur_Z, new_X):
        if cur_Z is None:
            return new_X[np.random.permutation(new_X.shape[0])[0:self.M], :]

        if self.use_old_Z:
            Z = np.copy(cur_Z)
        else:
            M_old = int(0.7 * self.M)
            M_new = self.M - M_old
            old_Z = cur_Z[np.random.permutation(self.M)[0:M_old], :]
            new_Z = new_X[np.random.permutation(new_X.shape[0])[0:M_new], :]
            Z = np.vstack((old_Z, new_Z))
        return Z

    def copy(self):
        temp = Streaming_Sparse_GP(self.initial_learning_kernel, self.M, use_old_Z=self.use_old_Z, alpha=self.alpha)
        temp._history["likelihood"] = self._history["likelihood"].copy()
        temp._history["kernel_params"] = copy.deepcopy(self._history["kernel_params"])
        if self.model is not None:
            temp.model = self.copy_model()
        return temp


    def _optimize(self, **options):
        best_model = None
        best_likelihood = -np.inf

        def fit():
            # fresh copy inside the fit so a jitter-retry does not restart from the
            # half-optimized (broken) state
            model = self.copy_model()

            # Fix inducing points
            #gpflow.set_trainable(model.inducing_variable, False)
            # Make inducing points trainable
            gpflow.set_trainable(model.inducing_variable, True)

            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(model.training_loss, model.trainable_variables, options=options)

            if hasattr(model, 'log_marginal_likelihood'):
                likelihood = model.log_marginal_likelihood().numpy()
            elif hasattr(model, 'training_loss'):
                likelihood = -model.training_loss().numpy()
            elif hasattr(model, 'log_posterior_density') and hasattr(model, 'log_prior_density'):
                likelihood = (model.log_posterior_density() - model.log_prior_density()).numpy()
            else:
                raise ValueError("can't get he likelihood")
            return model, likelihood

        for _ in range(self.attempts):
            model, likelihood = self._optimize_with_jitter(fit)

            if best_likelihood <= likelihood:
                best_model = model
                best_likelihood = likelihood

        self.model = best_model

    def create(self, X, y):
        Z = self.init_Z(None, X)

        self.model = SGPR_PEP((X, y), self.initial_learning_kernel, Z=Z,
                              alpha=self.alpha)
        self.model.likelihood.variance.assign(0.001)

        self._optimize(disp=1)

    def update(self, X, y):

        if self.model is None:
            self.create(X, y)
        else:
            Zopt = self.model.inducing_variable.Z.numpy()
            Zinit = self.init_Z(Zopt, X)

            mu, Su = self.model.predict_f(Zopt, full_cov=True)
            if len(Su.shape) == 3:
                Su = Su[0, :, :]

            Kaa = self.model.kernel(Zopt)

            variance = self.model.likelihood.variance

            self.model = OSGPR_PEP((X, y), self.copy_kernel(self.model.kernel),
                                mu, Su, Kaa, Zopt, Zinit, self.alpha)

            np_variance = variance.numpy()
            factor = [1.0,1.001, 1.01,1.1,2]
            success = False
            for f in factor:
                try:
                    self.model.likelihood.variance.assign(f*np_variance)
                    success = True
                    break
                except Exception:
                    print(f"Fail with variance: {f*np_variance} (bump factor: {f})")
            while not success:
                try:
                    self.model.likelihood.variance.assign(np_variance)
                    break
                except Exception as ee:
                    np_variance *= 10
                    print(f"Retrying with variance: {np_variance} (bump factor: 10)")
                    if np_variance >= 10:
                        raise ee


            self._optimize(disp=1)

        model = self.model
        if hasattr(model, 'model'):
            model = model.model

        if hasattr(model, 'log_marginal_likelihood'):
            likelihood = model.log_marginal_likelihood().numpy()
        elif hasattr(model, 'training_loss'):
            likelihood = -model.training_loss().numpy()
        elif hasattr(model, 'log_posterior_density') and hasattr(model, 'log_prior_density'):
            likelihood = (model.log_posterior_density() - model.log_prior_density()).numpy()
        else:
            raise ValueError("can't get he likelihood")

        self._history["likelihood"].append(likelihood)
        self._history["kernel_params"].append(
            {
                "kernel_name": str(type(model.kernel)),
                "kernel_param": {k: v.numpy() for k,v in parameter_dict(model.kernel).items()},
                # "mu":
            }
        )

