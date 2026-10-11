import numpy as np
from scipy.stats import multivariate_normal

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


def _to_numpy(x):
    if _HAS_TF and isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def _extract_points_and_times(X):
    """Return (points_at_t0, unique_times) from a (T*N, features+1) array."""
    times = X[:, -1]
    unique_times = np.unique(times)
    t0 = unique_times.min()
    mask = np.abs(times - t0) < 1e-10
    points = X[mask][:, :-1]
    return points, unique_times


def create_dist_from_model(model, X):
    """Build a Multivariate_Gaussian at the given query points X.

    Uses copy_posterior_mean to evaluate the gaussian process posterior mean as a pure
    numpy operation (one linear solve on construction, then a kernel product
    per call) rather than routing through GPflow's predict_f machinery.
    """
    kernel = model.copy_kernel(model.model.kernel)
    posterior_mean = model.copy_posterior_mean()
    X = _to_numpy(X)
    mu = posterior_mean(X).flatten()
    points, unique_times = _extract_points_and_times(X)
    return Multivariate_Gaussian(mu, kernel, points=points, time_series=unique_times)


def create_dist_from_model_no_predict(model):
    """
    Create a Multivariate_Gaussian from a trained gaussian process without any prediction call.

    Reads X_train and y_train directly from model.model.data and uses the
    observed y values as the mean vector.  No forward pass is run.

    This is the zero-cost alternative to create_dist_from_model(model, X):
    use it when the distribution is only needed at the training sensor locations.
    For predictions at new/arbitrary locations use create_dist_from_model instead.

    Note for Streaming_Sparse_GP: model.model.data holds the most recent batch
    only, so the returned distribution covers that batch's time range.
    """
    X_train, y_train = model.get_training_data()
    return create_dist_from_training_data(model, X_train, y_train)


def create_dist_from_training_data(model, X_train, y_train, get_times=True):
    """
    Build a Multivariate_Gaussian without running a gaussian process prediction pass.

    Uses the observed y_train values directly as the mean vector.  This is
    exact for a noise-free gaussian process (posterior mean at training points == data) and
    avoids the cost of a forward prediction when the test points coincide with
    the training points (e.g. Synthetic sensor locations).

    X_train: (T*N, spatial_features + 1), where the last column is time
    y_train: (true negative,) or (true negative, 1)
    """
    kernel = model.copy_kernel(model.model.kernel)
    X = _to_numpy(X_train)
    y = _to_numpy(y_train)
    points, unique_times = _extract_points_and_times(X)
    return Multivariate_Gaussian(y.flatten(), kernel, points=points, time_series=unique_times)

class Multivariate_Gaussian:
    def __init__(self, mu, kernel, points = None, time_series=None, *, subset=slice(None)):
        self._mu = np.array(mu)
        self.kernel = kernel
        self.points = points
        self.subset = subset
        self.time_series = time_series

    def update_points(self, points, time_series=None):
        self.points = points
        self.time_series = time_series

    def get_indice(self,A):
        def slice_to_indices(s, size):
            return np.arange(size)[s]
        if isinstance(A, slice):
            A = slice_to_indices(A, len(self.points))
        return A

    def get_subset(self, A=None, time_series=None):
        """Flat indices into the (time, space) mean vector, row-major over time."""
        if A is None:
            A = self.subset
        A = self.get_indice(A)

        if time_series is None:
            time_series = self.time_series

        if time_series is None:
            return A

        new_subset = []
        for i in time_series:
            for a in A:
                new_subset.append(len(self.points)*int(i) + a)
        return new_subset

    def get_points(self, A=None, time_series=None):

        # If kernel is a gpflow kernel
        if callable(self.kernel):
            pass
        # If kernel is a precomputed ndarray
        elif isinstance(self.kernel, np.ndarray):
            # Use np.ix_ to select rows/columns
            return self.get_subset(A, time_series), False
        else:
            raise ValueError("kernel must be a gpflow kernel or a precomputed ndarray")

        if A is None:
            A = self.subset
        A = self.get_indice(A)

        if time_series is None:
            time_series = self.time_series

        if time_series is None:
            return self.points[A], True

        new_subset = np.zeros((len(A)*len(time_series), self.points.shape[1] + 1))
        for i, t in enumerate(time_series):
            for j, a in enumerate(A):
                new_subset[len(A)*i + j] = np.append(self.points[a], t)
        return new_subset, True

    def mu(self, A=None, time_series=None):
        return self._mu[self.get_subset(A, time_series)]

    def Sigma(self, A=None, B=None, time_series_A=None, time_series_B=None):
        A, _ = self.get_points(A, time_series_A)
        B, do_gpflow = self.get_points(B, time_series_B)

        # If kernel is a gpflow kernel
        if do_gpflow:
            return self.kernel(A, B).numpy()
        # If kernel is a precomputed ndarray
        else:
            # Use np.ix_ to select rows/columns
            return self.kernel[np.ix_(A, B)]

    def sample(self, rnd=None):
        if rnd is not None:
            return rnd.multivariate_normal(self.mu(), self.Sigma())
        return np.random.multivariate_normal(self.mu(), self.Sigma())

    def __call__(self, A=None, time_series=None):
        if A is None:
            A = self.subset
        if time_series is None:
            time_series = self.time_series
        return Multivariate_Gaussian(self._mu, self.kernel, self.points, time_series=time_series, subset=A)

    def log_prob(self, data, time_series=None):
        if self.time_series is None:
            time_series = None
        return multivariate_normal.logpdf(data, mean=self.mu(time_series=time_series), cov=self.Sigma(time_series_A = time_series, time_series_B = time_series))
    def prob(self, data, time_series=None):
        if self.time_series is None:
            time_series = None
        return multivariate_normal.pdf(data, mean=self.mu(time_series=time_series), cov=self.Sigma(time_series_A = time_series, time_series_B = time_series))

    def marginalize_kernel(self, A=None, B=None, time_series=None):
        """Marginal spatial covariance over time: (1/T) sum_t K[(x,t),(x',t)].

        Only the synchronous (t == t') blocks are summed. Averaging the full
        (time x space) x (time x space) matrix over both time axes.
        The default time_series=[0] assumes the
        spatial covariance is the same at every instant, which holds for the separable
        kernels used here.
        """
        if self.time_series is None:
            return self.Sigma(A, B)

        if A is None:
            A = self.subset
        A = self.get_indice(A)
        if B is None:
            B = self.subset
        B = self.get_indice(B)
        if time_series is None:
            time_series = [0] # if assume it is the same at each time or else: self.time_series

        # only the t == t' blocks are needed
        K = np.zeros((len(A), len(B)))
        for t in time_series:
            K += self.Sigma(A, B, [t], [t])
        return K / len(time_series)
    def marginalize_mu(self, A=None, time_series=None):
        if self.time_series is None:
            return self.mu(A)

        n_space, n_time = len(self.points), len(self.time_series)


        if A is None:
            A = self.subset
        A = self.get_indice(A)
        if time_series is None:
            time_series = self.time_series

        if A is not None:
            n_space = len(A)
        if time_series is not None:
            n_time = len(time_series)

        # reshape into 2D tensor: (time, space)
        mu_reshaped = self.mu(A, time_series).reshape(n_time, n_space)

        # average over both time axes
        mu_space = mu_reshaped.mean(axis=(0))

        return mu_space
