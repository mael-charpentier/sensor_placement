import copy

import numpy as np
from gpflow.utilities import parameter_dict


class Extrapolation:
    def __init__(self, kernel, attempts=1):
        self.initial_learning_kernel = kernel
        self.attempts = attempts
        self.model = None
        self._history = {
            'means': [],
            'variances': [],
            'likelihood': [],
            'kernel_params': [],
        }

    def create(self, X, y):
        raise NotImplementedError

    def update(self, X, y):
        raise NotImplementedError

    def _optimize_with_jitter(self, fit_fn, max_jitter=1e1, default_jitte=1e-6, factor=10.0, max_tries=8):
        """Run one gaussian process fit attempt with numerical-failure recovery: when tensorflow aborts on a
        singular / non-positive semi definite matrix ("Input matrix is not invertible", "Cholesky
        decomposition was not successful"), raise GPflow's default jitter xfactor and
        retry fit_fn (which must build a fresh model, so the retry does not restart
        from the broken state).  Returns fit_fn's result as soon as the jitter is enough.        re-raises once max_jitter / max_tries is exhausted."""
        import gpflow
        import tensorflow as tf

        gpflow.config.set_default_jitter(default_jitte)

        for attempt in range(max_tries):
            try:
                return fit_fn()
            except (tf.errors.InvalidArgumentError, np.linalg.LinAlgError) as e:
                jitter = float(gpflow.config.default_jitter())
                if attempt == max_tries - 1 or jitter >= max_jitter:
                    raise
                new_jitter = min(max(jitter, 1e-8) * factor, max_jitter)
                msg = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
                print(f"  [warn] GP fit failed on a singular matrix ({msg}); "
                      f"raising GPflow jitter {jitter:g} -> {new_jitter:g} and retrying")
                gpflow.config.set_default_jitter(new_jitter)

    def predict_f(self, pred_x):
        mean, var = self.model.predict_f(pred_x, full_cov=False)
        return mean, var

    def copy(self):
        raise NotImplementedError

    def copy_model(self):
        # For GPflow models, deepcopy works but we should ensure no shared state
        new_model = copy.deepcopy(self.model)

        # Optional: reset optimizer state if any
        if hasattr(new_model, 'optimizer'):
            new_model.optimizer = None

        return new_model

    def copy_posterior_mean(self):
        """Return the gaussian process posterior mean as a standalone callable.

        Analogous to copy_kernel: extracts the learned parameters once so the
        returned function can be evaluated at any X without touching the GPflow
        model or running predict_f.

        Returned callable signature: f(X) -> np.ndarray of shape (len(X), 1)

        How it works:
        The posterior mean of a gaussian process is:
            mu(X) = k(X*, X_train) @ alpha
        where alpha = (K_train + sigma²I)⁻¹ y_train.

        alpha is computed once here (one linear solve, same cost as a single
        predict_f call).  Subsequent evaluations cost only a kernel product.

        Limitation: valid for Exact_GP only.  Sparse / streaming models
        do not store the full training data and require predict_f instead.
        """
        X_train, y_train = self.get_training_data()   # numpy arrays
        kernel = self.copy_kernel(self.model.kernel)
        sigma2 = float(self.model.likelihood.variance.numpy())

        import tensorflow as tf
        X_tf = tf.constant(X_train, dtype=tf.float64)
        K = kernel(X_tf, X_tf).numpy()
        K += sigma2 * np.eye(len(K))
        alpha = np.linalg.solve(K, y_train)           # (N, 1)

        # capture kernel, X_train, alpha by closure, no model reference kept
        def posterior_mean(X_new, batch_size=1000):
            import tensorflow as tf
            X_new = X_new.numpy() if hasattr(X_new, "numpy") else np.asarray(X_new)
            results = []
            for i in range(0, len(X_new), batch_size):
                X_batch = tf.constant(X_new[i:i + batch_size], dtype=tf.float64)
                K_batch = kernel(X_batch, X_tf).numpy()  # (batch, N)
                results.append(K_batch @ alpha)           # (batch, 1)
            return np.vstack(results)

        return posterior_mean

    def copy_kernel(self, kernel):
        # 1. Get the kernel class
        kernel_cls = type(kernel)

        # 2. Read parameter values as dict
        params = parameter_dict(kernel)
        param_values = {}
        for n, val in params.items():
            name = n.split(".")[-1]
            param_values[name] = val.numpy()

        # 3. Create new kernel with same parameters
        new_kernel = kernel_cls(**param_values)

        return new_kernel

    def predict_in_batches_with_var(self, X, batch_size=1000):
        mus, stds = [], []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            mu, var = self.model.predict_f(X_batch, full_cov=False)
            mu  = mu.numpy()  if hasattr(mu,  'numpy') else np.asarray(mu)
            var = var.numpy() if hasattr(var, 'numpy') else np.asarray(var)
            mus.append(mu)
            stds.append(np.sqrt(np.maximum(var, 0.0)))
        return np.vstack(mus), np.vstack(stds)

    def get_training_data(self):
        X, y = self.model.data
        X = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        return X, y

    def predict_in_batches_mu(self, X, batch_size=1000):
        mus = []


        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            mu, _ = self.predict_f(X_batch)
            mus.append(mu.numpy())
            #vars_.append(var.numpy)

        return np.vstack(mus) #, np.vstack(vars_)

    def predict(self, pred_x):
        mx, Sx = self.predict_f(pred_x)

        mx = mx.numpy() if hasattr(mx, "numpy") else mx
        Sx = Sx.numpy() if hasattr(Sx, "numpy") else Sx

        mx = mx.flatten()
        Sx = Sx.flatten()

        self._history['means'].extend(mx)
        self._history['variances'].extend(Sx)

        return mx, Sx

    @staticmethod
    def _kernel_info(obj):
        from spgprcpd.data.gaussian import Multivariate_Gaussian
        kernel = obj.kernel if isinstance(obj, Multivariate_Gaussian) else obj
        try:
            params = {k: v.numpy() for k, v in parameter_dict(kernel).items()}
        except Exception:
            params = None
        return {"kernel_name": str(type(kernel)), "kernel_param": params}

    def eval(self, true_data, kernel_prior, kernel_post, *, data=None, mask=None):
        """Score the accumulated predictions against true_data (element-wise mean squared error and
        95% confidence interval coverage).  mask (bool, flattened like true_data): score only the True
        entries, pass the observation mask of a real dataset so gap-filled values are
        never treated as ground truth.  n_eval reports how many points were scored."""
        if data is None:
            data = self._history
        result = {}

        means = np.array(data['means'], dtype=float)
        stds = np.sqrt(np.array(data['variances'], dtype=float))
        y_flat = np.asarray(true_data, dtype=float).flatten()
        n = min(len(y_flat), len(means))
        y_flat, means, stds = y_flat[:n], means[:n], stds[:n]
        if mask is not None:
            mask = np.asarray(mask, dtype=bool).flatten()[:n]
            y_flat, means, stds = y_flat[mask], means[mask], stds[mask]

        result['n_eval'] = int(len(y_flat))
        if len(y_flat):
            result['MSE'] = float(np.mean((y_flat - means) ** 2))
            # Proportion of test points within the 95% confidence interval around the predicted mean
            lower = means - 1.96 * stds
            upper = means + 1.96 * stds
            result['coverage_95'] = float(np.mean((y_flat >= lower) & (y_flat <= upper)))
        else:
            result['MSE'] = float('nan')
            result['coverage_95'] = float('nan')

        #spearman_corr, _ = spearmanr(y_flat, means)
        #result['spearman'] = spearman_corr

        #pearson_corr, _ = pearsonr(y_flat, means)
        #result['pearson'] = pearson_corr

        result['likelihood'] = copy.deepcopy(self._history["likelihood"])
        result['kernel_params'] = copy.deepcopy(self._history["kernel_params"])

        result['true_kernel_prior'] = self._kernel_info(kernel_prior)
        result['true_kernel_post'] = {"kernel_name": None, "kernel_param": None}
        if kernel_post is not None:
            result['true_kernel_post'] = self._kernel_info(kernel_post)

        return result


