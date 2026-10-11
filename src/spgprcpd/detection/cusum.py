import numpy as np

from spgprcpd.detection.base import CPD


class CuSum(CPD):
    _supports_calibration = True

    def __init__(self, dist_prior, dist_post, threshold = None, alpha_arl = None):
        super().__init__(threshold, alpha_arl, detect_only_change=False)
        self.dist_prior = dist_prior
        self.dist_post = dist_post
        self._get_threshold()

        self.last_value = None

    def _rebuild_with_threshold(self, threshold):
        return CuSum(self.dist_prior, self.dist_post, threshold=threshold)

    def _default_bracket(self):
        return (0.0, 15.0)

    @property
    def ARL(self):
        return np.exp(self.threshold)

    @property
    def EDD(self):
        mu0 = self.dist_prior.mu()
        mu1 = self.dist_post.mu()
        Sigma0 = self.dist_prior.Sigma()
        Sigma1 = self.dist_post.Sigma()

        k = len(mu0)

        diff = mu1 - mu0

        logdet0 = np.linalg.slogdet(Sigma0)[1]
        logdet1 = np.linalg.slogdet(Sigma1)[1]

        term_logdet = logdet0 - logdet1
        term_quad = diff.T @ np.linalg.solve(Sigma0, diff)
        term_trace = np.trace(np.linalg.solve(Sigma0, Sigma1))

        kl = 0.5 * (term_logdet - k + term_quad + term_trace)

        return self.threshold / kl

    def _find_threshold(self):
        return np.log(self.alpha_arl)

    def get_all_params(self):
        return {
            "threshold": self.threshold,
        }

    def reset(self):
        super().reset()
        self.last_value = None

    def _values(self, unselect_set, selected_set, data, time_already):
        dist_prior = self.dist_prior(A = selected_set)
        dist_post = self.dist_post(A = selected_set)

        g = 0 if self.last_value is None else self.last_value

        value_g = []

        for t in range(len(data)):
            ll0 = dist_prior.log_prob(data[t], [time_already + t])
            ll1 = dist_post.log_prob(data[t], [time_already + t])

            g = max(0, g + (ll1-ll0))
            value_g.append(g)

        self.last_value = g

        return value_g


class CuSum_Without_Post(CPD):
    _supports_calibration = True

    def __init__(self, dist_prior, dist_post, k=0, threshold = None, alpha_arl = None, calibrate_k=0.1):
        super().__init__(threshold, alpha_arl, detect_only_change=False)
        self.dist_prior = dist_prior
        self.dist_post = dist_post
        self.k = k
        self.calibrate_k=calibrate_k
        self._get_threshold()

        self.last_value = None

    def _rebuild_with_threshold(self, threshold):
        return CuSum_Without_Post(self.dist_prior, self.dist_post, k=self.k, threshold=threshold)

    def _default_bracket(self):
        return (0.0, 500.0)

    @property
    def ARL(self):
        raise NotImplementedError

    @property
    def EDD(self):
        raise NotImplementedError(
            "EDD for CuSum_Without_Post requires knowing the expected shift delta; "
            "subclass and override this property."
        )

    def _find_threshold(self):
        return np.log(self.alpha_arl)

    def reset(self):
        super().reset()
        self.last_value = None

    def get_all_params(self):
        return {
            "threshold": self.threshold,
            "k": self.k,
        }

    def _alarms(self, unselect_set, selected_set, values, l):
        alarms = [i for i in range(len(values)) if any([v > self.threshold for v in values[i]])]
        return alarms

    def _values(self, unselect_set, selected_set, data, time_already):
        dist_prior = self.dist_prior(A = selected_set)
        mu = dist_prior.marginalize_mu()
        Sigma = dist_prior.marginalize_kernel()

        value_g = []

        if isinstance(Sigma, np.ndarray) and Sigma.shape[0] == 1:
            Sigma = Sigma.item()

        if isinstance(Sigma, np.ndarray):
            # Multivariate case: the increment is the squared
            # Mahalanobis distance from the pre-change mean, Q_t = (x-mu0)^T Sigma0^-1 (x-mu0),
            # used as a single one-sided statistic g = Q_t - alpha (allowance self.k = alpha >= the placement size).
            C = 0 if self.last_value is None else self.last_value
            for t in range(len(data)):
                diff = data[t] - mu
                q = diff.T @ np.linalg.solve(Sigma, diff)
                C = max(0, C + q - self.k)
                value_g.append([C])
            self.last_value = C
        else:
            if isinstance(mu, np.ndarray):
                mu = mu.item()
            # Univariate case: standardized deviation z = (x-mu0)/sigma0,
            # two one-sided statistics accumulating positive and negative shifts.
            C_plus, C_minus = (0, 0) if self.last_value is None else self.last_value
            for t in range(len(data)):
                d = data[t]
                if isinstance(d, np.ndarray):
                    d = d.item()
                z = (d - mu) / Sigma
                C_plus = max(0, C_plus + z - self.k)
                C_minus = max(0, C_minus - z - self.k)
                value_g.append([C_plus, C_minus])
            self.last_value = (C_plus, C_minus)

        return value_g


    def calibrate_on_f0(self, unselect_set, selected_set, target_arl=None, *,
                        f0_data=None, n_streams=40, horizon=None, seed=0):

        if self.calibrate_k is not None:
            self.k = len(selected_set) + 5 * self.calibrate_k

        return super().calibrate_on_f0(unselect_set, selected_set, target_arl,
                        f0_data=f0_data, n_streams=n_streams, horizon=horizon, seed=seed)

    def calibrate_on_f0_f1(self, unselect_set, selected_set, target_arl=None, *,
                           f0_data=None, chg_data=None, tau=None, W=20,
                           n_streams=40, horizon=None, seed=0, n_grid=12, grid_tol=0.5):

        if self.calibrate_k is not None:
            prior = self.dist_prior(selected_set)
            post = self.dist_post(selected_set)

            mu0 = prior.marginalize_mu()
            Sigma0 = prior.marginalize_kernel()

            mu1 = post.marginalize_mu()
            Sigma1 = post.marginalize_kernel()

            diff = mu1 - mu0
            trace_term = np.trace(np.linalg.solve(Sigma0, Sigma1))
            quad_term = diff @ np.linalg.solve(Sigma0, diff)
            upper_bound= trace_term + quad_term

            self.k = self.calibrate_k * len(selected_set) + (1-self.calibrate_k) * upper_bound

        return super().calibrate_on_f0_f1(unselect_set, selected_set, target_arl,
                        f0_data=f0_data, chg_data=chg_data, tau=tau, W=W,
                        n_streams=n_streams, horizon=horizon, seed=seed, n_grid=n_grid, grid_tol=grid_tol)
