from itertools import combinations

import numpy as np


class CPD:
    """Base class for all CPD algorithms.

    This class contains the shared code for all CPD algorithms.

    Parameters:
    threshold: float, optional
        The threshold for the algorithm. If not provided, optimize the model.
    alpha_arl: float, optional
        The parameter for the ARL in the optimization model.

    Attributes:
    ARL: float
        The result of the ARL for this CPD algorithm.
    expected detection delay: float
        The result of the expected detection delay for this CPD algorithm.
    __history: dict
        A dictionary containing the history of the algorithm.
        It contains the following keys: 'values', 'alarms'.
    """
    # whether this detector supports data-driven threshold calibration from f0 alone.
    # A subclass that implements _rebuild_with_threshold sets it to True
    _supports_calibration = False

    def __init__(self, threshold = None, alpha_arl = None, *, detect_only_change = True):
        self.threshold = threshold
        self.alpha_arl = alpha_arl

        self.detect_only_change = detect_only_change

        # set by _get_threshold when the threshold is left to data-driven calibration. The
        # calibration then runs on the first alarms call, once the placement is known.
        self._deferred_calibration = False

        self.__history = {
            'values': [],
            'alarms': []
        }

    @property
    def ARL(self):
        raise NotImplementedError

    @property
    def EDD(self):
        raise NotImplementedError

    def _find_threshold(self):
        raise NotImplementedError

    def reset(self):
        self.__history = {
            'values': [],
            'alarms': []
        }


    def _get_threshold(self):
        if self.threshold is None:
            if self.alpha_arl is None:
                raise ValueError("threshold or alpha_arl must be set")
            #self.threshold = self._find_threshold
            if self._can_calibrate():
                self._deferred_calibration = True

    def _can_calibrate(self):
        return (self._supports_calibration
                and self.alpha_arl is not None
                and getattr(self, "dist_prior", None) is not None)

    def _default_bracket(self):
        """(lo, hi) bracket for the threshold search. Overridden per detector."""
        return (0.0, 10.0)

    def _calib_expand(self):
        """Whether the upper bracket may grow geometrically (False for bounded thresholds
        such as the bocpd alarm probability)."""
        return True

    def _f0_streams(self, f0_data, selected_set, n_streams, horizon, rnd):
        """f0 calibration streams. With a real historical record f0_data: the first stream
        is that record, and any further streams (n_streams > 1) are sampled from the
        historical distribution self.dist_prior, so n_streams=1 uses the historical
        stream alone, and larger n_streams adds model-sampled streams to reduce the arl0
        estimate's variance. Without f0_data, all streams are sampled from dist_prior."""
        from spgprcpd.detection.calibration import simulate_f0_streams
        if f0_data is None:
            return simulate_f0_streams(self.dist_prior, selected_set, n_streams, horizon, rnd)
        f0_data = np.asarray(f0_data)
        if n_streams <= 1:
            return [f0_data]
        return [f0_data] + simulate_f0_streams(self.dist_prior, selected_set,
                                               n_streams - 1, f0_data.shape[0], rnd)

    def _chg_streams(self, chg_data, selected_set, n_streams, horizon, tau, rnd):
        """f0 to f1 change streams for the expected detection delay in calibrate_on_f0_f1. With a real change
        record chg_data (its change at tau): the first stream is that record, and any
        further streams (n_streams > 1) are sampled from dist_prior -> dist_post
        (change at tau), so n_streams=1 uses the historical change stream alone.
        Without chg_data, all streams are sampled."""
        from spgprcpd.detection.calibration import simulate_change_streams
        if chg_data is None:
            return simulate_change_streams(self.dist_prior, self.dist_post, selected_set,
                                           n_streams, horizon, tau, rnd)
        chg_data = np.asarray(chg_data)
        if n_streams <= 1:
            return [chg_data]
        return [chg_data] + simulate_change_streams(self.dist_prior, self.dist_post, selected_set,
                                                    n_streams - 1, chg_data.shape[0], tau, rnd)

    def calibrate_on_f0(self, unselect_set, selected_set, target_arl=None, *,
                        f0_data=None, n_streams=40, horizon=None, seed=0):
        """Calibrate the threshold on f0 alone, by bisecting it until the empirical arl0
        reaches target_arl, which defaults to self.alpha_arl.

        Pass f0_data, a real in-control series of shape (T, n_sensors), to calibrate on it by
        moving-block bootstrap, so the threshold matches the distribution of the real
        statistic. Without it, in-control streams are simulated from self.dist_prior instead.
        Sets and returns self.threshold, and does nothing when the detector cannot be
        calibrated."""
        if not self._can_calibrate():
            return self.threshold
        from spgprcpd.detection.calibration import calibrate_f0
        target = self.alpha_arl if target_arl is None else target_arl
        # streams must be long enough to observe run lengths up to the target arl0, otherwise
        # the empirical arl0 is censored at the horizon and cannot reach the target.
        if horizon is None:
            horizon = max(300, int(2 * target))
        rnd = np.random.RandomState(seed)
        f0 = self._f0_streams(f0_data, selected_set, n_streams, horizon, rnd)
        # a target above the stream length is unreachable, the run length gets censored at
        # the horizon first, so clip it and calibrate against what can actually be observed
        target = min(target, f0[0].shape[0] * 0.9)
        self.threshold = calibrate_f0(self._rebuild_with_threshold, unselect_set, selected_set,
                                      f0, target, self._default_bracket(),
                                      expand=self._calib_expand())
        self._deferred_calibration = False
        self.reset()
        return self.threshold

    def calibrate_on_f0_f1(self, unselect_set, selected_set, target_arl=None, *,
                           f0_data=None, chg_data=None, tau=None, W=20,
                           n_streams=40, horizon=None, seed=0, n_grid=12, grid_tol=0.5):
        """Calibrate on f0 and f1. It builds a grid of thresholds out of the f0 bisection
        bracket of calibrate_on_f0, taken with return_bracket, stopped early at the coarser
        grid_tol and discretised uniformly. It then picks the threshold minimising delta on
        the f0 to f1 streams, delta being the expected detection delay divided by W plus the
        false alarm rate, with W the detection window the caller asked for. Falls back to
        calibrate_on_f0 when there is no post-change distribution. Sets and returns
        self.threshold, and leaves the grid curve of (theta, delta, edd, far) in
        self._calib_curve."""
        if not self._can_calibrate() or getattr(self, "dist_post", None) is None:
            return self.calibrate_on_f0(unselect_set, selected_set, target_arl,
                                        f0_data=f0_data, n_streams=n_streams,
                                        horizon=horizon, seed=seed)
        from spgprcpd.detection.calibration import calibrate_f0, calibrate_f0_f1
        target = self.alpha_arl if target_arl is None else target_arl
        if horizon is None:
            horizon = max(300, int(2 * target))
        if tau is None:
            tau = (int(np.asarray(chg_data).shape[0]) // 2) if chg_data is not None else (horizon // 2)
        rnd = np.random.RandomState(seed)
        f0 = self._f0_streams(f0_data, selected_set, n_streams, horizon, rnd)
        # a target above the stream length is unreachable, the run length gets censored at
        # the horizon first, so clip it and calibrate against what can actually be observed
        target = min(target, f0[0].shape[0] * 0.9)
        chg = self._chg_streams(chg_data, selected_set, n_streams, horizon, tau, rnd)
        # the grid is the final bracket of the f0 bisection, stopped early with the coarser
        # tolerance grid_tol, then discretised uniformly
        lo_b, hi_b = calibrate_f0(self._rebuild_with_threshold, unselect_set, selected_set,
                                  f0, target, self._default_bracket(), tol=grid_tol,
                                  expand=self._calib_expand(), return_bracket=True)
        self.reset()
        grid = np.linspace(lo_b, hi_b, n_grid)
        # pick theta_star by delta, the expected detection delay over W plus the false alarm
        # rate, measured on the f0 to f1 streams
        theta_star, best_curve, curve = calibrate_f0_f1(self._rebuild_with_threshold, unselect_set,
                                            selected_set, chg, tau, grid, W)

        self._calib_curve = best_curve
        self.threshold = float(theta_star)
        self._deferred_calibration = False
        self.reset()
        return self.threshold

    def _values(self, unselect_set, selected_set, data, l):
        raise NotImplementedError

    def _alarms(self, unselect_set, selected_set, values, l):
        alarms = [i for i in range(len(values)) if values[i] > self.threshold]
        return alarms

    def alarms(self, unselect_set, selected_set, data, time_local=True):
        """Return the alarms for the given data.

        Parameters:
        unselect_set: list
            The set of points that are not selected.
        selected_set: list
            The set of points that are selected.
        data: dict
            A dictionary containing the data collected by the selected points.

        Returns:
        alarms: list
            A list of indices in the data where where have been a change.
        """

        if self._deferred_calibration:
            self.calibrate_on_f0(unselect_set, selected_set)

        l = len(self.__history['values'])

        values = self._values(unselect_set, selected_set, data, l)
        alarms = self._alarms(unselect_set, selected_set, values, l)

        global_time_alarms = list(map(lambda x: x+l, alarms))

        self.__history['values'].extend(values)
        self.__history['alarms'].extend(global_time_alarms)
        return alarms if time_local else global_time_alarms


    def __RI_change_points_to_labels(self, n, change_points):
        labels = np.array([0] * n)
        cp = sorted(change_points)

        current_label = 0
        prev_cp = 0

        for c in cp:
            labels[prev_cp:c] = current_label
            current_label += 1
            prev_cp = c

        labels[prev_cp:] = current_label
        return labels

    def __match_events(self, true_points, pred_points, tol=5):
        """Point-event precision / recall / F1.

        A true change point is matched to at most one alarm within +-tol, and an alarm
        is spent by at most one true change point: the matching is one-to-one, so precision
        and recall count the same matched pairs and both lie in [0, 1].
        """
        true_points = sorted(set(true_points))
        pred_points = sorted(set(pred_points))

        matched_true = set()
        matched_pred = set()

        for t in true_points:
            for p in pred_points:
                if p in matched_pred:
                    continue
                if abs(t - p) <= tol:
                    matched_true.add(t)
                    matched_pred.add(p)
                    break

        # fraction of the alarms that are true positives / of the true change points covered
        precision = len(matched_pred) / len(pred_points) if pred_points else 0.0
        recall = len(matched_true) / len(true_points) if true_points else 0.0

        f1 = (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)

        return precision, recall, f1

    def __sustained_scores(self, n, tau, pred_points):
        """Per-time-step precision / recall / F1 for a detector meant to stay in alarm
        through the post-change regime (the caveat on the true detection rate).

        Every alarm at t >= tau is a true positive, every alarm at t < tau a false
        positive, and every post-change step left unflagged a false negative, the same
        split the false alarm rate and the true detection rate make, summarized into one number.  The recall is the true detection rate.        the precision, the share of the alarms that fall after the change, is what the true detection rate
        alone does not say.
        """
        pred_points = sorted(set(pred_points))

        tp = len([t for t in pred_points if t >= tau])
        fp = len(pred_points) - tp
        fn = max(n - tau + 1, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)

        return precision, recall, f1



    def __RI_score(self, n, true_points, pred_points):
        if n > 1000:
            return np.nan

        true_labels = self.__RI_change_points_to_labels(n, true_points)
        pred_labels = self.__RI_change_points_to_labels(n, pred_points)

        RI_agree = 0
        RI_total = 0

        for i, j in combinations(range(n), 2):
            same_true = (true_labels[i] == true_labels[j])
            same_pred = (pred_labels[i] == pred_labels[j])

            if same_true == same_pred:
                RI_agree += 1
            RI_total += 1
        return RI_agree / RI_total if RI_total != 0 else None

    def get_alarm_history(self):
        """Return all global alarm time-indices accumulated across every alarms call."""
        return list(self.__history['alarms'])

    def get_value_history(self):
        """Return the full CPD statistic series accumulated across every alarms call."""
        return list(self.__history['values'])

    def eval(self, true_change, *, data=None):
        """Evaluate the performance of the CPD algorithm.

        Parameters:
        true_change: int or None
            The true change point.  None means the stream is assumed to contain no
            change (the common case on real data, where no event is annotated): every
            alarm is then a false alarm and the detector is perfect iff it stays silent
            (no_alarm True). The delay metrics are nan.
        data: dict, optional
            A dictionary containing the history of the algorithm.
            If not provided, the history of the algorithm will be used.

        Returns:
        result: dict
            A dictionary containing the performance metrics of the algorithm
            (num_alarms / no_alarm report the raised alarms in both cases).
        """
        if data is None:
            data = self.__history

        n = len(data['values'])

        if true_change is None:
            pred_points = data['alarms']
            detection_delays_first = [v  for v in pred_points]
            no_alarm = len(pred_points) == 0
            if self.detect_only_change:
                false_alarm_rate = len(pred_points)
            else:
                false_alarm_rate = len(pred_points) / max(n, 1)
            return {
                'detection_delay_first': detection_delays_first,
                'detection_delay': np.nan,
                'false_alarm_rate': false_alarm_rate,
                'true_alarm_rate': np.nan,
                'num_alarms': len(pred_points),
                'no_alarm': no_alarm,
                'RI': self.__RI_score(n, [], pred_points),
                'F1': {
                    'score': 1.0 if no_alarm else 0.0,
                    'precision': 1.0 if no_alarm else 0.0,
                    'recall': 1.0 if no_alarm else 0.0,
                },
                'F1_sustained': {
                    'score': 1.0 if no_alarm else 0.0,
                    'precision': 1.0 if no_alarm else 0.0,
                    'recall': 1.0 if no_alarm else 0.0,
                },
            }

        tau = true_change + 1
        pred_points = [v + 1 for v in data['alarms']]

        RI_score = self.__RI_score(n, [true_change], data['alarms'])

        precision, recall, f1 = self.__match_events([tau], pred_points, tol=5)
        precision_s, recall_s, f1_s = self.__sustained_scores(n, tau, pred_points)

        detection_delays_first = [t for t in pred_points]
        detection_delays = [t for t in pred_points if t >= tau]

        if self.detect_only_change:
            false_alarm_rate = len([t for t in pred_points if t < tau])
            true_alarm_rate = len(detection_delays)
        else:
            false_alarm_rate = len([t for t in pred_points if t < tau]) / max(tau - 1, 1)
            true_alarm_rate = len(detection_delays) / max(n - tau + 1, 1)

        result = {
            'detection_delay_first': min(detection_delays_first) - tau + 1 if detection_delays_first else np.nan,
            'detection_delay': min(detection_delays) - tau + 1 if detection_delays else  np.nan,
            'false_alarm_rate': false_alarm_rate,
            'true_alarm_rate': true_alarm_rate,
            'num_alarms': len(pred_points),
            'no_alarm': len(pred_points) == 0,
            'RI': RI_score,
            'F1': {
                'score': f1,
                'precision': precision,
                'recall': recall,
            },
            'F1_sustained': {
                'score': f1_s,
                'precision': precision_s,
                'recall': recall_s,
            },
        }

        return result


