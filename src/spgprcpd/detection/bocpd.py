import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t

from spgprcpd.detection.base import CPD


class BOCPD(CPD):
    _supports_calibration = True

    def __init__(self, dist_prior, hazard_lambda=lambda r : 0, threshold = None, alpha_arl = None, dist_post=None, k=0):
        super().__init__(threshold, alpha_arl, detect_only_change=False)
        self.dist_prior = dist_prior
        self.dist_post = dist_post
        self.hazard_lambda = hazard_lambda
        self.last_log_B = np.log(np.array([1.0]))
        self._get_threshold()
        self.data_history_values = None
        # Detection window k. _values returns the alarm statistic
        # P(r_t <= k | x_{1:t}) = probability a change occurred within the last k steps, and the
        # alarm is P(r_t <= k) > h. The ideal statistic P(r_t = 0) is degenerate under a constant
        # hazard, the shared one-step predictive cancels in the normalisation, leaving
        # P(r_t = 0) == H regardless of the data, so detection needs k >= 1.
        self.k = k
        # maximum a posteriori run length argmax_r P(r_t=r | x_{1:t}) per step, also populated by _values. It
        # collapses to approx. 0 at a change and is the alternative bocpd detection signal.
        self.last_map_run_length = None

    def _rebuild_with_threshold(self, threshold):
        return BOCPD(self.dist_prior, hazard_lambda=self.hazard_lambda, threshold=threshold,
                     dist_post=self.dist_post, k=self.k)

    def _default_bracket(self):
        return (0.0, 1.0)     # alarm threshold is a probability P(r_t <= k)

    def _calib_expand(self):
        return False          # bounded in (0, 1). Do not grow the bracket past 1

    def get_H(self):  # assume constant
        return self.hazard_lambda(1)

    @property
    def ARL(self):
        raise NotImplementedError

    @property
    def EDD(self):
        raise NotImplementedError

    def _find_threshold(self):
        raise NotImplementedError

    def reset(self):
        super().reset()
        self.last_log_B = np.log(np.array([1.0]))
        self.last_map_run_length = None
        self.data_history_values = None

    def get_all_params(self):
        return {
            "threshold": self.threshold,
            "k": self.k,
            "hazard": self.hazard_lambda,
        }

    def _niw_predictive_logpdf(self, x_t, segment, mu0, Sigma0):
        """One-step posterior-predictive log-density of a Normal-Inverse-Wishart segment
        model. Segment holds the observations already gathered
        in the current run (shape (n, d), or None when the run is empty).        conditioning on them, the predictive of x_t is a multivariate Student-t
        (for d=1, its normal inverse wishart generalisation for d>1).
        """
        x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
        d = x_t.shape[0]
        mu0 = np.atleast_1d(np.asarray(mu0, dtype=float)).reshape(d)
        Sigma0 = np.atleast_2d(np.asarray(Sigma0, dtype=float)).reshape(d, d)

        # Weakly-informative normal inverse wishart prior anchored on the pre-change distribution (mu0, Sigma0)
        kappa0 = 1.0
        nu0 = d + 2.0
        Psi0 = Sigma0

        if segment is None:
            seg = np.empty((0, d))
        else:
            seg = np.asarray(segment, dtype=float).reshape(-1, d)
            seg = seg[~np.isnan(seg).any(axis=1)]
        n = seg.shape[0]

        if n == 0:
            kappa_n, nu_n, mu_n, Psi_n = kappa0, nu0, mu0, Psi0
        else:
            # conjugate updates
            xbar = seg.mean(axis=0)
            kappa_n = kappa0 + n
            nu_n = nu0 + n
            mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
            S = (seg - xbar).T @ (seg - xbar)
            diff = (xbar - mu0).reshape(d, 1)
            Psi_n = Psi0 + S + (kappa0 * n / kappa_n) * (diff @ diff.T)

        # multivariate Student-t predictive
        dof = nu_n - d + 1.0
        scale = (kappa_n + 1.0) / (kappa_n * dof) * Psi_n
        scale = 0.5 * (scale + scale.T) + 1e-9 * np.eye(d)
        return multivariate_t.logpdf(x_t, loc=mu_n, shape=scale, df=dof)

    def _values(self, unselect_set, selected_set, data, time_already):

        # build joint Gaussian over [past, current]
        dist = self.dist_prior(A=selected_set)
        mu0 = dist.marginalize_mu()         # pre-change mean over the sensor set
        Sigma0 = dist.marginalize_kernel()  # pre-change covariance over the sensor set

        if self.data_history_values is None:
            self.data_history_values = data.copy()
        else:
            self.data_history_values = np.vstack([self.data_history_values, data])
        def pred_fn(x_t, r_prev, t):
            t = t + time_already

            if dist.time_series is None:
                # i.i.d.-in-time segment model: the one-step predictive is the
                # Normal-Inverse-Wishart conjugate Student-t,
                # conditioned on the r_prev observations gathered since the last change.
                segment = self.data_history_values[t-r_prev:t] if r_prev > 0 else None
                return self._niw_predictive_logpdf(x_t, segment, mu0, Sigma0)

            # A temporal kernel is available
            if r_prev == 0:
                return dist.log_prob(x_t, [t])

            # indices of past data in this run
            past_idx = np.arange(t - r_prev, t)

            # Mean
            mu_p = dist.mu(time_series=past_idx)
            mu_t = dist.mu(time_series=[t])

            # Covariances
            K_pp = dist.Sigma(time_series_A=past_idx, time_series_B=past_idx)
            K_pt = dist.Sigma(time_series_A=past_idx, time_series_B=[t])
            K_tp = dist.Sigma(time_series_A=[t], time_series_B=past_idx)
            K_tt = dist.Sigma(time_series_A=[t], time_series_B=[t])

            # observed past values
            x_p = self.data_history_values[past_idx]

            # conditional mean
            K_pp = K_pp + 1e-6*np.eye(K_pp.shape[0])
            mu_cond = mu_t + K_tp @ np.linalg.solve(K_pp, x_p.reshape(len(past_idx) * len(x_t)) - mu_p)

            # conditional covariance
            Sigma_cond = K_tt - K_tp @ np.linalg.solve(K_pp, K_pt)

            return multivariate_normal.logpdf(x_t, mean=mu_cond.ravel(), cov=Sigma_cond)


        T = len(data)
        eps = 1e-12

        # B[r] = joint probability P(r_t = r, x_1:t)

        # R[t, r] = P(r_t = r | x_1:t)
        # R = np.zeros((T, T + 1))
        # alarm statistic P(r_t <= k): probability a change occurred within the last k steps
        cp_probs = np.zeros(T)
        map_rl = np.zeros(T)   # maximum a posteriori run length argmax_r P(r_t=r): the detection signal (see __init__)

        for t in range(T):
            x_t = data[t]

            run_length = len(self.last_log_B)

            log_B_new = np.full(run_length + 1, -np.inf)

            # growth probabilities + Changepoint probability
            cp = []
            for r_prev in range(run_length):
                r = r_prev + 1
                H = np.clip(self.hazard_lambda(r_prev), eps, 1 - eps)
                log_B_new[r] = (
                    np.log(1-H)
                    + pred_fn(x_t, r_prev, t)
                    + self.last_log_B[r_prev]
                )

                cp.append(np.log(H) + pred_fn(x_t, r_prev, t) + self.last_log_B[r_prev])

            log_B_new[0] = logsumexp(cp)

            # normalize
            Z = logsumexp(log_B_new)
            log_B_new -= Z

            # store posterior
            # R[t,:t+2] = B_new[:t+2]
            # store only what matters
            cp_probs[t] = logsumexp(log_B_new[:self.k + 1])   # log P(r_t <= k | x_{1:t})
            map_rl[t] = int(np.argmax(log_B_new))

            self.last_log_B = log_B_new

        self.last_map_run_length = map_rl
        return np.exp(cp_probs) # P(r_t <= k)
