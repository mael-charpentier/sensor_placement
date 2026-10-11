import numpy as np

from spgprcpd.detection import bocpd
from spgprcpd.detection.base import CPD


class EWMA(CPD):
    _supports_calibration = True

    def __init__(self, alpha, dist_prior, dist_post, threshold = None, alpha_arl = None):
        super().__init__(threshold, alpha_arl, detect_only_change=False)
        self.alpha = alpha
        self.dist_prior = dist_prior
        self.dist_post = dist_post
        self._get_threshold()

        self.last_value = None

        self.mu = lambda sensors : self.dist_prior.marginalize_mu(sensors)

    def _rebuild_with_threshold(self, threshold):
        return EWMA(self.alpha, self.dist_prior, self.dist_post, threshold=threshold)

    def _default_bracket(self):
        return (1.0, 500.0)

    def SigmaZ(self, sensors, t):
        factor   = (self.alpha / (2 - self.alpha)) * (1 - (1 - self.alpha)**(2 * t))
        Sigma_AA = self.dist_prior.marginalize_kernel(A=sensors, B=sensors)   # the placement sizexthe placement size
        if len(sensors) == 1:
            return np.sqrt(Sigma_AA * factor)   # scalar std  sigma_{S_t}   (univariate)
        n = len(sensors)
        SigmaZ = Sigma_AA * factor
        # marginalize_kernel can be near-singular (highly correlated sensors), so the whitening
        # the statistic covariance^{-1} in Z = S^T the statistic covariance^{-1} S blows up. A small ridge (a fraction of the mean
        # diagonal) keeps it well-conditioned.
        return SigmaZ + 1e-2 * np.mean(np.diag(SigmaZ)) * np.eye(n)

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
        self.last_value = None

    def get_all_params(self):
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
        }


    def _alarms(self, unselect_set, selected_set, values, l):
        alarms = []
        if len(selected_set) == 1:
            # univariate ewma: alarm when S_t leaves [mu0 +- L * sigma_{S_t}]
            mu = float(self.mu(selected_set)[0])
            for i in range(len(values)):
                s  = float(np.ravel(values[i][0])[0])
                sd = float(np.ravel(self.SigmaZ(selected_set, l + i + 1))[0])   # sigma_{S_t} (std)
                if not (mu - self.threshold * sd < s < mu + self.threshold * sd):
                    alarms.append(i)
        else:
            # multivariate ewma: alarm when Z_t = S_t^T the statistic covariance^{-1} S_t >= h
            for i in range(len(values)):
                if values[i][0] >= self.threshold:
                    alarms.append(i)
        return alarms

    def _values(self, unselect_set, selected_set, data, time_already, weight = None):

        if self.last_value is not None:
            g = self.last_value
        elif len(selected_set) == 1:
            g = self.mu(selected_set)
        else:
            g = 0

        if weight is None:
            weight = np.full(len(data), self.alpha)

        value_g = []

        for t in range(len(data)):
            d = data[t] - (0 if len(selected_set) == 1 else self.mu(selected_set))
            g = (1-weight[t])*g + weight[t]*d
            if len(selected_set) != 1:
                # multivariate ewma: g = S_t^T the statistic covariance^{-1} S_t
                S = np.ravel(g)
                Z = S @ np.linalg.solve(self.SigmaZ(selected_set, time_already + t + 1), S)
                value_g.append([float(Z), g])
            else:
                value_g.append([float(np.ravel(g)[0])])

        self.last_value = g

        return value_g


class EWMA_BOCPD(EWMA):
    _supports_calibration = False

    def __init__(self, alpha, dist_prior, dist_post, hazard_lambda=lambda r : 0, k=5, threshold = None, alpha_arl = None):
        self.bocpd = bocpd.BOCPD(dist_prior, hazard_lambda, threshold, alpha_arl, k=k)
        super().__init__(alpha, dist_prior, dist_post, threshold, alpha_arl)
        self._get_threshold()

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
        self.bocpd.reset()

    def get_all_params(self):
        return {
            "ewma": super().get_all_params(),
            "bocpd": self.bocpd.get_all_params(),
        }

    def _values(self, unselect_set, selected_set, data, time_already):

        weight = self.bocpd._values(unselect_set, selected_set, data, time_already)
        return super()._values(unselect_set, selected_set, data, time_already, weight=weight)
