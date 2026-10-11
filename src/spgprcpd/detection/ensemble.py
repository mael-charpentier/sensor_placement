"""Per-location ensemble detector, the monitor each sensor independently option.

Instead of one multivariate detector on the joint observation vector, Multiple_CPD runs an
independent copy of a base detector at each monitored sensor, each on that sensor's own
one-dimensional stream, and each skipping only its own nan steps, so a gap at one sensor
never affects the others. The system raises an alarm at a time step when enough of the
per-location detectors alarm there:

  * mode="any" needs 1 sensor, so it alarms if any location fires.
  * mode="majority" needs floor(n/2) + 1, so it alarms if most locations fire.
  * mode=k, an int, needs k sensors.

The ensemble is itself a spgprcpd.detection.base.CPD, so it plugs into the same pipeline and
the same data-driven calibration. One shared threshold is calibrated until the system arl0
reaches the target, through the usual split between _rebuild_with_threshold, _values for the
threshold-free statistic and _alarms for the threshold-dependent decision, which is what lets
spgprcpd.detection.calibration.calibrate_f0 compute each statistic once and then only
re-threshold.
"""
import numpy as np

from spgprcpd.detection.base import CPD


class Multiple_CPD(CPD):
    """Ensemble of one univariate base detector per monitored sensor.

    Parameters:
    sub_factory: callable(dist_prior, dist_post, *, threshold, alpha_arl) -> CPD
        Builds one fresh per-location base detector.  Wrap the base class so that alpha, k
        and the like are already fixed, as in
        lambda p, q, **kw: CuSum_Without_Post(p, q, k=0.5, **kw) or
        lambda p, q, **kw: EWMA(0.1, p, q, **kw).
    dist_prior, dist_post: Multivariate_Gaussian
        Shared laws. Each sub-detector marginalises them to its own sensor (A=[sensor]).
    mode: {"any", "majority"} or int
        System alarm rule (see the module docstring).
    threshold, alpha_arl: float, optional
        The shared threshold / arl0 budget of the whole ensemble (one calibrated value).
    """
    _supports_calibration = True

    def __init__(self, sub_factory, dist_prior, dist_post, *, mode="any",
                 threshold=None, alpha_arl=None):
        super().__init__(threshold, alpha_arl, detect_only_change=False)
        self.sub_factory = sub_factory
        self.dist_prior = dist_prior
        self.dist_post = dist_post
        self.mode = mode
        # a prototype sub-detector (never fed data): source of the calibration bracket and of
        # get_all_params, and it fixes whether the ensemble supports calibration at all.
        self._proto = sub_factory(dist_prior, dist_post, threshold=threshold,
                                  alpha_arl=alpha_arl)
        self._supports_calibration = getattr(self._proto, "_supports_calibration", False)
        self.subs = None          # one base detector per sensor (lazy: needs the placement size)
        self._sensors = None
        self._obs_counts = None   # per-sensor count of observed (non-nan) steps so far
        self._get_threshold()

    # plumbing
    def _ensure_subs(self, selected_set):
        """(Re)build one fresh sub-detector per sensor with the current shared threshold."""
        if self.subs is None or self._sensors != list(selected_set):
            self._sensors = list(selected_set)
            self.subs = [self.sub_factory(self.dist_prior, self.dist_post,
                                          threshold=self.threshold, alpha_arl=self.alpha_arl)
                         for _ in selected_set]
            self._obs_counts = [0] * len(selected_set)

    def _rebuild_with_threshold(self, threshold):
        return Multiple_CPD(self.sub_factory, self.dist_prior, self.dist_post,
                            mode=self.mode, threshold=threshold, alpha_arl=self.alpha_arl)

    def _default_bracket(self):
        return self._proto._default_bracket()

    def _calib_expand(self):
        return self._proto._calib_expand()

    def reset(self):
        super().reset()
        self.subs = None
        self._sensors = None
        self._obs_counts = None

    def _n_required(self, n):
        """Number of locations that must fire together for a system alarm."""
        if self.mode == "majority":
            return n // 2 + 1
        if isinstance(self.mode, (int, np.integer)) and not isinstance(self.mode, bool):
            return int(np.clip(self.mode, 1, n))
        return 1                                   # "any" (default)

    def get_all_params(self):
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "n_locations": None if self._sensors is None else len(self._sensors),
            "sub": self._proto.get_all_params(),
        }

    # statistic (threshold-free)
    def _values(self, unselect_set, selected_set, data, time_already):
        """Per-location statistic series.  Returns a length-T list. Entry t is a list
        over the sensors of that sensor's native base-detector value at t (whatever the
        base _values yields, a scalar, [C], [C+, C-].), or None where the
        sensor is unobserved (nan).  Each sub-detector sees only its observed steps, so a nan
        at one sensor is skipped for that sensor alone."""
        self._ensure_subs(selected_set)
        arr = np.asarray(data, float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        T, n = arr.shape
        values = [[None] * n for _ in range(T)]
        for jj, sensor in enumerate(selected_set):
            for k, t in enumerate(arr[:, jj]):
                if not np.isnan(t):
                    values[k][jj] = self.subs[jj]._values(unselect_set, [sensor], [t],
                                           self._obs_counts[jj])[0]
                    self._obs_counts[jj] += 1
        return values

    # alarm decision (thresholded)
    def _alarms(self, unselect_set, selected_set, values, l):
        """Aggregate the per-location alarms of the base detector into system alarms: a step
        is a system alarm when at least _n_required locations alarm there.  Re-derives each
        location's alarms from its stored statistic series (skipping its nan steps), so a
        candidate threshold is applied without recomputing the statistic (calibration)."""
        self._ensure_subs(selected_set)
        n = len(selected_set)
        T = len(values)
        counts = [0] * T
        for jj, sensor in enumerate(selected_set):
            times, series = [], []
            for t in range(T):
                v = values[t][jj] if jj < len(values[t]) else None
                if v is not None:
                    times.append(t)
                    series.append(v)
            if not series:
                continue
            # the base rule is stateless in the threshold. Run it on this sensor's observed
            # series (local step index), then map the local alarms back to original time.
            local = self.subs[jj]._alarms(unselect_set, [sensor], series, 0)
            for k in local:
                counts[times[k]] += 1
        need = self._n_required(n)
        return [t for t in range(T) if counts[t] >= need]

    def get_value_history(self):
        """Per-location statistic history flattened for plotting: one column per sensor, its
        representative scalar (first component of the native value) at each step, nan where the
        sensor is unobserved.  (The base structure is nested. This gives a plottable array.)"""
        raw = super().get_value_history()
        if not raw:
            return raw
        n = max((len(row) for row in raw), default=0)
        out = []
        for row in raw:
            step = []
            for jj in range(n):
                v = row[jj] if jj < len(row) else None
                if v is None:
                    step.append(float("nan"))
                elif isinstance(v, (list, tuple, np.ndarray)):
                    step.append(float(np.ravel(v)[0]))
                else:
                    step.append(float(v))
            out.append(step)
        return out
