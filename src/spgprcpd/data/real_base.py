"""Shared base for data managers built on a real sensor record.

Both real datasets, the EcoSense Montreal deployment and the Intel Berkeley Lab network, are
exposed through the same interface as spgprcpd.data.synthetic.Synthetic_Data_Manager, so the
placement and detection pipeline runs unchanged on real data. A subclass only has to load
its record and call super().__init__ with

  * coords, the (n, 2) sensor coordinates in a local metric frame.
  * Y, the (T, n) dense observation block, one column per sensor on a regular time grid
    with no nan in it, see densify.
  * time_index, optional, length T and datetime64, so change times can be given as
    timestamps instead of bin indices.

The base then produces the same objects as the synthetic manager:

  * locations, V, shape (n, 2), real coordinates in metres, re-centred.
  * dataset, X and y and y_iid, shapes (T, n, d) and (T, n), where y is the observed level
    and y_iid its first difference, the stream the detectors consume.
  * distribution, dist_prior, an empirical Multivariate_Gaussian over the sensors estimated
    from the historical record.

A real historical record only shows the normal regime f0. Neither of these two datasets
contains two occurrences of the same change, so there is nothing to estimate a second regime
from and dist_post is always None. The record splits at split_time into the historical phase,
indices 0 up to split, used for estimation and calibration, and the test phase, split up to
T. change_time is optional, it is the timestamp of a known event annotated in the test
stream, or None when the stream is assumed change-free, in which case the detectors are
scored on staying quiet through cpd.eval(None).

get_dataset slices the two phases through the overridden method below.
"""
import numpy as np

from spgprcpd.data.gaussian import Multivariate_Gaussian
from spgprcpd.data.synthetic import Synthetic_Data_Manager


def kernel_gaussian(block, coords, mask, extrapolation_cls, kernel_init):
    """Spatial Multivariate_Gaussian whose covariance comes from a kernel fitted to the
    historical block, the estimator, as opposed to the entrywise
    empirical_gaussian below.

    We fit on the observed entries only (mask), flattening (time, sensor) into the
    (x, y, t) rows the gaussian process expects, then read f0 off the fitted model with
    create_dist_from_model: its kernel gives the covariance and its posterior mean gives
    the mean at any location, including one absent from the block.
    """
    from spgprcpd.data.gaussian import create_dist_from_model

    block = np.asarray(block, float)
    coords = np.asarray(coords, float)
    T, n = block.shape
    mask = np.ones_like(block, bool) if mask is None else np.asarray(mask, bool)
    mask = mask & ~np.isnan(block)

    # (time, sensor) -> one (x, y, t) row per observed entry
    t_idx, s_idx = np.nonzero(mask)
    if len(t_idx) < 2:
        raise ValueError("not enough observed entries to fit f^0")
    X_train = np.hstack([coords[s_idx], t_idx.reshape(-1, 1).astype(float)])
    y_train = block[t_idx, s_idx].reshape(-1, 1)

    if len(X_train.shape) == 2:
        x_mean = np.nanmean(X_train, axis=0)
        x_std = np.nanstd(X_train, axis=0)
    elif len(X_train.shape) == 3:
        x_mean = np.nanmean(X_train, axis=(0, 1))
        x_std = np.nanstd(X_train, axis=(0, 1))
    x_std[x_std == 0] = 1.0

    y_mean = np.nanmean(y_train)
    y_std = np.nanstd(y_train)
    if y_std == 0:
        y_std = 1.0

    # Normalize the concatenated data
    X_train = (X_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std

    gp = extrapolation_cls(kernel_init)
    gp.update(X_train, y_train)

    # every location at t = 0 -> a purely spatial f0 over the n sensors
    X_query = np.hstack([coords, np.zeros((n, 1))])

    if len(X_query.shape) == 2:
        x_mean = np.nanmean(X_query, axis=0)
        x_std = np.nanstd(X_query, axis=0)
    elif len(X_query.shape) == 3:
        x_mean = np.nanmean(X_query, axis=(0, 1))
        x_std = np.nanstd(X_query, axis=(0, 1))
    x_std[x_std == 0] = 1.0

    X_query = (X_query - x_mean) / x_std
    return create_dist_from_model(gp, X_query)


def empirical_gaussian(block, coords, ridge=1e-6, mask=None):
    """Empirical spatial Multivariate_Gaussian from a (T, n) block: mean over time,
    covariance across the n sensors, projected to the nearest positive semi definite matrix.
    time_series=None -> a purely spatial law, which the detectors and placement criteria
    use through marginalize_mu / marginalize_kernel.

    mask (True = actually observed): estimate only from existing data: interpolated /
    missing entries are excluded (pairwise-complete masked covariance).  Entries that cannot
    be estimated (a sensor with no observation in the block) fall back to mean 0 /
    covariance 0, kept valid by the ridge."""
    block = np.asarray(block, float)
    if mask is None and not np.isnan(block).any():
        mu = np.nanmean(block, axis=0)
        Sigma = np.atleast_2d(np.cov(block, rowvar=False))
    else:
        bad = np.isnan(block)
        if mask is not None:
            bad |= ~np.asarray(mask, bool)
        m = np.ma.masked_array(np.nan_to_num(block), mask=bad)
        mu = np.asarray(np.ma.filled(m.mean(axis=0), 0.0))
        Sigma = np.ma.cov(m, rowvar=False, allow_masked=True)
        Sigma = np.atleast_2d(np.asarray(np.ma.filled(Sigma, 0.0)))
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma)
    Sigma = (V * np.clip(w, 0.0, None)) @ V.T + ridge * np.eye(Sigma.shape[0])
    return Multivariate_Gaussian(mu, Sigma, points=np.asarray(coords, float), time_series=None)


def densify(M, time_index=None):
    """Fill internal NaNs per column by linear interpolation, then drop rows still nan
    (leading/trailing gaps).  Returns (M_dense, time_index_dense)."""
    M = np.array(M, float)
    idx = np.arange(M.shape[0])
    for j in range(M.shape[1]):
        col = M[:, j]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            M[:, j] = np.interp(idx, idx[ok], col[ok])
    keep = ~np.isnan(M).any(axis=1)
    if time_index is not None:
        time_index = np.asarray(time_index)[keep]
    return M[keep], time_index


def remove_flat_runs(M, min_run):
    """nan-out runs of >= min_run consecutive identical values per column (the
    signature of a dead/stuck sensor that keeps repeating its last reading)."""
    if min_run is None:
        return np.array(M, float)
    M = np.array(M, float)
    for j in range(M.shape[1]):
        col = M[:, j]
        prev = np.roll(col, 1)
        prev[0] = np.nan
        # a new run starts where the value differs from the previous one (nan never equals)
        new_run = (col != prev) | np.isnan(col) | np.isnan(prev)
        rid = np.cumsum(new_run)
        runlen = np.bincount(rid)[rid]
        col[(runlen >= min_run) & ~np.isnan(col)] = np.nan
        M[:, j] = col
    return M


def lonlat_to_meters(lat, lon):
    """Project (lat, lon) in decimal degrees to a local metric frame (equirectangular around
    the centroid), re-centred at zero.  Good to <1 m over a city-scale deployment."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat0 = np.nanmean(lat)
    R = 6_371_000.0
    x = np.deg2rad(lon - np.nanmean(lon)) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return np.column_stack([x, y])


class Real_Data_Manager_Base(Synthetic_Data_Manager):
    """Drop-in real-data replacement for Synthetic_Data_Manager (module docstring).

    Parameters:
    coords: (n, 2) array, sensor coordinates (local metres).
    Y: (T, n) array, observation block, columns laid out
        [candidates | test | historical-only].  nan is allowed anywhere (real per-sensor
        gaps, kept rather than interpolated): every consumer uses only the observed entries
        (observed), gaussian process inference trains on the observed points (get_dataset drops the
        nan (X, y) pairs) and is scored on observed entries, the per-location detectors of
        spgprcpd/detection/ensemble.py skip each sensor's own nan, and f0 / f1 are estimated from the
        masked covariance.  A candidate/test column may not be entirely nan (no data at all).
    N_test: int, held-out test locations (columns right after the
                                        candidates): only used to evaluate inference.
    N_hist_only: int, trailing columns used only for the historical
                                        estimation (never placeable, never evaluated).
    observed: (T, n) bool or None, True where the value was actually observed (False on
        interpolated / filled entries).  f0 / f1 are estimated from observed increments
        only, and the benchmarks evaluate inference on observed test entries only.
    no_hist_cols: iterable[int], column indices whose data is excluded from the
        historical f0 estimation (e.g. A candidate whose past record is untrusted:
        placeable and evaluable, but contributing no historical knowledge).
    split_time: int/timestamp or None, boundary between the historical phase [0, split)
        and the test phase [split, T). Default T/2.  Ints are bin indices. Anything else is
        located in time_index.
    change_time: int/timestamp or None, known/annotated real change in the test stream,
        or None (default) when no change is assumed there.
    max_samples: int or None, cap T by even subsampling (detection streams and
        calibration stay tractable on very long records).
    time_index: (T,) datetime64 or None, bin timestamps (kept in self.time_index).
    sensor_ids: list or None, external ids of the columns (kept in self.sensor_ids).
    """

    def __init__(self, coords, Y, *, N_test=0, N_hist_only=0, observed=None,
                 no_hist_cols=(), split_time=None, change_time=None,
                 max_samples=None, seed=0, time_index=None, sensor_ids=None,
                 extrapolation_cls=None, kernel_prior_init=None):
        Y = np.asarray(Y, float)
        coords = np.asarray(coords, float)
        if Y.ndim != 2 or coords.shape[0] != Y.shape[1]:
            raise ValueError(f"Y {Y.shape} and coords {coords.shape} are inconsistent")
        n_used = Y.shape[1] - N_hist_only          # candidate + test columns

        if Y.shape[0] and n_used and np.isnan(Y[:, :n_used]).all(axis=0).any():
            raise ValueError("a candidate/test column of Y is entirely NaN (no observation); "
                             "drop that sensor or widen the time range")
        observed = (~np.isnan(Y) if observed is None
                    else np.asarray(observed, bool))
        if observed.shape != Y.shape:
            raise ValueError(f"observed {observed.shape} and Y {Y.shape} are inconsistent")

        if max_samples is not None and Y.shape[0] > max_samples:
            idx = np.linspace(0, Y.shape[0] - 1, max_samples).astype(int)
            Y = Y[idx]
            observed = observed[idx]
            if time_index is not None:
                time_index = np.asarray(time_index)[idx]
        T, n = Y.shape
        if n <= N_test + N_hist_only:
            raise ValueError(f"only {n} sensors but N_test={N_test}, "
                             f"N_hist_only={N_hist_only}")

        self.time_index = None if time_index is None else np.asarray(time_index)
        self.sensor_ids = list(sensor_ids) if sensor_ids is not None else list(range(n))
        self.observed = observed

        # phase split: f0-only historical record [0, split), then the test stream
        # change_time is optional: a known real event in the test stream, else None
        split_time = T // 2 if split_time is None else self._to_index(split_time)
        change_time = self._to_index(change_time)
        if not (0 < split_time < T):
            raise ValueError(f"invalid split_time={split_time} for T={T}")
        if change_time is not None and not (split_time < change_time <= T):
            raise ValueError(f"change_time={change_time} must lie in the test phase "
                             f"({split_time}, {T}]")

        # attributes expected by the inherited get_dataset / criteria
        # layout [candidates | test | hist-only]: get_dataset's only_testing_loc==1 picks
        # exactly the N_test columns after the candidates, so the hist-only columns are never
        # selected nor evaluated, they only enter the (masked) f0 / f1 estimation below.
        self.num_samples = T
        self.num_points = n - N_test - N_hist_only
        self.N_test = N_test
        self.N_hist_only = N_hist_only
        self.num_features = coords.shape[1]
        self.grid = float(np.max(np.abs(coords))) if coords.size else 10.0
        self.seed = seed
        self.seeds_point = None
        self.random_walk = 1                 # y is a level series. Y_iid is its first difference
        self.noise_std = 0.0
        self.do_estimation = False
        self.only_testing_loc = 1 if N_test > 0 else True
        self.norm_saved = []

        self.split_time = split_time
        self.change_time = change_time
        self.hist_change_times = (split_time, split_time)
        self.test_change_time = None if change_time is None else change_time - split_time

        # locations + tensors, laid out exactly like the synthetic manager
        self.V = coords                                              # (n, features)
        self.X = np.broadcast_to(coords, (T, n, self.num_features)).copy()
        self.y = Y                                                   # (T, n) observed level
        self.y_iid = np.vstack([Y[:1], np.diff(Y, axis=0)])          # first differences

        # f0 from the whole historical phase (f0-only,)
        # The detectors and placement criteria consume the first-difference series y_iid, so
        # f0 is estimated from y_iid, not the raw level.  Estimation uses only observed
        # increments (both endpoints actually observed): interpolated / filled / missing
        # entries are masked out.  Real data never exposes f1 -> dist_post is None. Only
        # the f0-only criteria and detectors apply.
        Yd = self.y_iid
        self.no_hist_cols = sorted(int(c) for c in no_hist_cols)
        est_obs = observed
        if self.no_hist_cols:
            # columns whose past is untrusted contribute no historical knowledge
            est_obs = observed.copy()
            est_obs[:, self.no_hist_cols] = False
        obs_inc = np.vstack([est_obs[:1], est_obs[1:] & est_obs[:-1]])
        self.dist_prior = self._estimate_prior(Yd, coords, obs_inc, split_time,
                                               extrapolation_cls, kernel_prior_init)
        self.dist_post = None
        # aliases the parent uses internally
        self._dist_prior = self.dist_prior
        self._dist_post = None

    def _estimate_prior(self, Yd, coords, obs_inc, split_time,
                        extrapolation_cls, kernel_prior_init):
        """f0 over the candidate locations, from the historical increments.

        Prefers the fitted-kernel estimator (kernel_gaussian), and
        falls back on the entrywise empirical one only when it cannot run.  The fallback is not equivalent: it
        leaves a zero row for any sensor missing from the historical block, so such a sensor
        cannot be ranked by an informed criterion.
        """
        hist = slice(1, split_time)

        if extrapolation_cls is None or kernel_prior_init is None:
            print("[warn] no estimation model given: f^0 falls back to the entrywise "
                  "empirical covariance; sensors absent from the historical record will "
                  "get a zero row and cannot be ranked by an informed criterion")
            return empirical_gaussian(Yd[hist], coords, mask=obs_inc[hist])
        try:
            return kernel_gaussian(Yd[hist], coords, obs_inc[hist],
                                   extrapolation_cls, kernel_prior_init)
        except ImportError as e:
            print(f"[warn] cannot fit f^0 ({e.__class__.__name__}: {e}); falling back to "
                  f"the entrywise empirical covariance -- sensors absent from the "
                  f"historical record cannot be ranked")
            return empirical_gaussian(Yd[hist], coords, mask=obs_inc[hist])

    def get_dataset(self, A, B, time_dimension=True, normalize=True, batch_size=None,
                    time_range=slice(None), time_output=False, phase="test",
                    save_norm=False):
        """Same contract as the parent, but the historical/test phase slicing is done here:
        the parent only slices phases when a post-change distribution exists, and real data
        has f0 only (dist_post is None)."""
        if time_range == slice(None) or time_range is None:
            time_range = (slice(0, self.split_time) if phase == "historical"
                          else slice(self.split_time, self.num_samples))
        return super().get_dataset(A, B, time_dimension=time_dimension,
                                   normalize=normalize, batch_size=batch_size,
                                   time_range=time_range, time_output=time_output,
                                   phase=phase, save_norm=save_norm)

    def _to_index(self, t):
        if t is None or isinstance(t, (int, np.integer)):
            return t
        if self.time_index is None:
            raise ValueError(f"change time {t!r} is a timestamp but no time_index is available")
        ts = np.datetime64(str(t).replace(" ", "T") if isinstance(t, str) else t)
        return int(np.searchsorted(self.time_index.astype("datetime64[s]"),
                                   ts.astype("datetime64[s]")))

    def observed_mask(self, cols, phase="test"):
        rng = (slice(0, self.split_time) if phase == "historical"
               else slice(self.split_time, self.num_samples))
        return self.observed[rng][:, list(cols)]

    def describe(self):
        span = ""
        if self.time_index is not None:
            span = f", {self.time_index[0]} -> {self.time_index[-1]}"
        k, t, h = self.num_points, self.N_test, self.N_hist_only
        eval_extra = getattr(self, "eval_extra", [])
        groups = (f"placement={self.sensor_ids[:k]}, "
                  f"evaluation={sorted(self.sensor_ids[k:k + t] + [self.sensor_ids[c] for c in eval_extra])}"
                  + (f", hist-only={self.sensor_ids[k + t:]}" if h else "")
                  + (f", no-hist={[self.sensor_ids[c] for c in self.no_hist_cols]}"
                     if self.no_hist_cols else ""))
        change = ("no change assumed in the test stream (detectors scored on staying silent)"
                  if self.change_time is None else
                  f"known change at t={self.change_time} (t={self.test_change_time} of the "
                  f"test stream)")
        return (f"{k} candidate + {t} test" + (f" + {h} hist-only" if h else "")
                + f" sensors, T={self.num_samples}{span}; f0-only historical phase "
                f"[0, {self.split_time}), test phase [{self.split_time}, {self.num_samples}); "
                f"{change}; {groups}")
