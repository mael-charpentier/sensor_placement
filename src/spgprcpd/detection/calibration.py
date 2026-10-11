"""Data driven calibration of the detector thresholds.

Two procedures, depending on what is known about the data.

calibrate_f0 needs only in control streams. It bisects on the threshold until the
empirical average run length under f0 reaches a target, which is a false alarm budget.
This works for every detector, including the ones with no closed form run length such as
segcpgp and bocpd.

calibrate_f0_f1 also uses the post change law. It scores each threshold on a grid by
delta = edd / tau + far, where edd is the expected detection delay and far the false alarm
rate, measured on streams that switch from f0 to f1, and returns the minimiser. The grid
is the bracket that calibrate_f0 ends up with, discretised uniformly.

Both take a make_detector(threshold) factory that returns a fresh detector with its shape
parameters already fixed, so a stateful detector is never reused across streams. Streams
are (T, n_sensors) arrays, which is exactly what CPD.alarms consumes. The two simulate_
helpers build such streams from an estimated f0 and f1, which is the simulate from the
fitted model option. The alternative is to pass block bootstrapped slices of the real
record instead, which is what the real data benchmarks do.
"""
import numpy as np


# stream generation (simulate from the fitted f0 / f1)
def _iid_from_marginal(d, n, rnd):
    """Draw n i.i.d. Steps from the time-marginalised spatial law of d."""
    mu = np.ravel(d.marginalize_mu())
    Sig = np.atleast_2d(d.marginalize_kernel())
    return rnd.multivariate_normal(mu, Sig, size=n)


def simulate_f0_streams(dist0, selected_set, n_streams, T, rnd=None):
    """Draw n_streams in-control streams of length T at selected_set from f0, each
    step an i.i.d. Draw from the marginalised f0 law. Returns a list of (T, n_sensors) arrays."""
    rnd = np.random.RandomState(0) if rnd is None else rnd
    d0 = dist0(A=selected_set)
    return [_iid_from_marginal(d0, T, rnd) for _ in range(n_streams)]


def bootstrap_streams(series, n_streams, horizon, rnd=None):
    """Moving-block bootstrap of a real in-control series (T, n_sensors): draw n_streams
    contiguous windows of length horizon, preserving the local temporal dependence of
    the record."""
    rnd = np.random.RandomState(0) if rnd is None else rnd
    series = np.atleast_2d(np.asarray(series, float))
    if series.shape[0] == 1 and series.shape[1] != 1:
        series = series.T
    T = len(series)
    out = []
    for _ in range(n_streams):
        start = rnd.randint(0, max(T, 1))
        out.append(series[(start + np.arange(horizon)) % T])
    return out


def simulate_change_streams(dist0, dist1, selected_set, n_streams, T, tau, rnd=None):
    """Draw n_streams streams of length T that run under f0 and switch to f1 at
    tau (the labelled change), each step i.i.d. From the marginalised regime law (consistent
    with the detectors' model).  Returns a list of (T, n_sensors) arrays."""
    rnd = np.random.RandomState(0) if rnd is None else rnd
    d0, d1 = dist0(A=selected_set), dist1(A=selected_set)
    return [np.vstack([_iid_from_marginal(d0, tau, rnd), _iid_from_marginal(d1, T - tau, rnd)])
            for _ in range(n_streams)]


# empirical operating characteristics
def _first_alarm_time(detector, unselect_set, selected_set, stream):
    """Run length: the index of the first alarm, or the horizon if none fires."""
    alarms = detector.alarms(unselect_set, selected_set, stream, time_local=True)
    return (alarms[0] + 1) if alarms else len(stream)
def _mean_alarm_time(detector, unselect_set, selected_set, stream):
    """Run length: the (1-based) index of the first alarm, or the horizon if none fires."""
    alarms = detector.alarms(unselect_set, selected_set, stream, time_local=True)
    return len(stream)/(len(alarms) if alarms else 1)


def _values_per_stream(make_detector, unselect_set, selected_set, streams, theta0):
    """Compute each detector statistic series once per stream (the statistic is
    threshold-independent). A fresh detector per stream resets stateful detectors (e.g. Bocpd).
    The calibration then re-derives the alarms from these precomputed values for every candidate
    threshold via detector._alarms(values), never recomputing the (expensive) _values."""
    return [make_detector(theta0)._values(unselect_set, selected_set, s, 0) for s in streams]


def empirical_arl0(make_detector, threshold, unselect_set, selected_set, f0_streams):
    """Empirical arl0(threshold): mean time to the first alarm on in-control f0 streams."""
    rls = [
        #_first_alarm_time(make_detector(threshold), unselect_set, selected_set, s)
        _mean_alarm_time(make_detector(threshold), unselect_set, selected_set, s)
        for s in f0_streams
    ]
    return float(np.mean(rls))


def empirical_edd(make_detector, threshold, unselect_set, selected_set, change_streams, tau):
    """Empirical expected detection delay(threshold): mean of (T_D - tau + 1) over the f0 to f1 streams with
    T_D >= tau, the conditional detection delay."""
    delays = []
    for s in change_streams:
        alarms = make_detector(threshold).alarms(unselect_set, selected_set, s, time_local=True)
        after = [a for a in alarms if a >= tau]
        if after:
            delays.append(after[0] - tau + 1)
    return float(np.mean(delays)) if delays else np.nan


# calibration from f0 alone
def calibrate_f0(make_detector, unselect_set, selected_set, f0_streams, target_arl,
                 theta_bracket, tol=1e-3, max_iter=60, expand=True, return_bracket=False):
    """Calibrate the threshold on f0 to reach a target arl0 (false-alarm control).

    arl0 is monotone increasing in the threshold, so we bisect for the smallest threshold
    whose empirical arl0 reaches target_arl.

    Parameters:
    make_detector: callable(threshold) -> CPD
        Builds a fresh detector (shape parameters fixed) with the given threshold.
    f0_streams: list of (T, n_sensors) arrays
        In-control streams (bootstrap of the record, or simulate_f0_streams).
    theta_bracket: (lo, hi)
        Initial bracket. Hi is grown geometrically if it does not reach the target.
    return_bracket: bool
        If True, return the final search bracket (theta_lo, theta_hi) instead of the
        calibrated threshold, pass a coarser tol to stop the bisection early with a
        wider bracket, which calibrate_f0_f1 then discretises into its grid
.

    Returns:
    theta_star: float, or (theta_lo, theta_hi) when return_bracket is True
    """
    lo, hi = float(theta_bracket[0]), float(theta_bracket[1])

    # compute each statistic series once (threshold-independent). The search below only
    # re-derives the alarms from these precomputed values, never recomputing _values.
    vals = _values_per_stream(make_detector, unselect_set, selected_set, f0_streams, lo)

    def arl(theta):
        probe = make_detector(theta)
        ms = []
        for v in vals:
            a = probe._alarms(unselect_set, selected_set, v, 0)
            ms.append(len(v) / (len(a) if a else 1))       # mean-alarm-time arl0 estimator
        return float(np.mean(ms))

    if expand:
        prev = -1.0
        last_prev = -1.0
        it = 0
        last_hi = lo
        while it < 40:
            a = arl(hi)
            # stop once the target is reachable, or once arl0 has plateaued, e.g. It is
            # censored at the stream horizon (target_arl > horizon), so no larger threshold
            # can help. The caller should lengthen the streams to resolve a larger arl0.
            if a >= target_arl:
                break
            elif (a <= prev + 1e-9 and a <= last_prev + 1e-9):
                return (lo, hi) if return_bracket else hi
            last_prev = prev
            prev = a
            last_hi = hi
            hi = lo + 2.0 * (hi - lo)
            it += 1
        lo = last_hi

    for _ in range(max_iter):
        if hi - lo <= (tol if lo <= 0 else tol * lo):
            break
        mid = 0.5 * (lo + hi)
        if arl(mid) >= target_arl:
            hi = mid
        else:
            lo = mid
    return (lo, hi) if return_bracket else hi


# calibration from both f0 and f1
def calibrate_f0_f1(make_detector, unselect_set, selected_set, change_streams, tau, theta_grid, W):
    """Pick the threshold minimising the mean, over the f0 to f1 streams, of the per-stream
    score delta, which is the expected detection delay divided by W plus the false alarm
    rate, where W is the detection window.

    Per threshold, for each change stream, in control up to tau and f1 after it:

      * expected detection delay is T_D - tau + 1, with T_D the first alarm at or after tau,
        or 2W when the stream never detects. That cap censors the delay so that a
        non-detection costs a finite 2 rather than infinity.
      * false alarm rate is the number of alarms before tau over tau - 1.
      * delta is the expected detection delay over W, plus the false alarm rate.

    Time is 1-indexed inside the loop, the same convention as CPD.eval. The pre-change window
    is then the steps 1 to tau-1, which really holds tau-1 of them, and that is what the false
    alarm rate divides by. tau and the alarm indices arrive 0-indexed, so both are shifted
    once. theta_star minimises the mean per-stream delta over the grid by exhaustive search,
    since the grid is not monotone.

    The statistic is threshold-independent, so it is computed once per stream and only
    re-thresholded across the grid.  Returns (theta_star, best_curve, curve) with curve a list of
    (theta, [(delta, edd, far) for each stream]).
    """
    valsC = _values_per_stream(make_detector, unselect_set, selected_set, change_streams, theta_grid[0])
    tau = tau + 1
    denom = max(tau - 1, 1)

    curve = []
    for theta in theta_grid:
        probe = make_detector(theta)
        score = []
        for v in valsC:
            al = [i + 1 for i in probe._alarms(unselect_set, selected_set, v, 0)]
            after = [i for i in al if i >= tau]
            edd = np.inf # max(W, 1) * 2
            if after:
                edd = after[0] - tau + 1                      # expected detection delay = T_D - tau + 1
            far = (len([i for i in al if i < tau]) / denom)   # false alarm rate: false alarms before tau
            delta = edd / max(W, 1) + far
            score.append((delta, edd, far))

        curve.append((float(theta), score))

    best_curve = min(curve, key=lambda c: np.mean([cc[0] for cc in c[1]]))
    return best_curve[0], best_curve[1], curve
