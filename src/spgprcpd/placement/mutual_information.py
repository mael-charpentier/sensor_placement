import numpy as np

from spgprcpd.placement.base import Metric


class Mutual_Information(Metric):
    def __init__(self, *, type_algo = 0, percent_substitution = 0):
        super().__init__(type_algo = type_algo, percent_substitution=percent_substitution)

    def _cond_var_kernel(self, x, X_S, kernel):
        """Var(x | X_S) using kernel"""
        if X_S is None or len(X_S) == 0:
            return kernel(x, x).Sigma.squeeze()

        K_xx = kernel.marginalize_kernel(A=x, B=x).squeeze()
        K_xS = kernel.marginalize_kernel(A=x, B=X_S)          # (1, |S|)
        K_SS = kernel.marginalize_kernel(A=X_S, B=X_S)        # (|S|, |S|)

        result = K_xx - K_xS @ np.linalg.solve(K_SS, K_xS.T)
        if result.shape == (1,1):
            return result.squeeze()
        return result

    def _MI_set(self, A, B, dist_prior):
        if len(A) == 0 or len(B) == 0:
            return 0.0

        A = np.array(A)
        B = np.array(B)

        Sigma_A = dist_prior.marginalize_kernel(A, A)
        Sigma_B = dist_prior.marginalize_kernel(B, B)
        AB = np.concatenate([A, B])
        Sigma_AB = dist_prior.marginalize_kernel(AB, AB)

        _, det_A = np.linalg.slogdet(Sigma_A)
        _, det_B = np.linalg.slogdet(Sigma_B)
        _, det_AB = np.linalg.slogdet(Sigma_AB)

        return 0.5 * (det_A + det_B - det_AB)

    def _MI_add_point(self, A, C, x, dist_prior):
        # 1) Compute Var(x | A)
        if len(A) == 0:
            # No conditioning yet
            var_A = dist_prior.marginalize_kernel(x, x).squeeze()
        else:
            var_A = self._cond_var_kernel(x, A, dist_prior)

        # 2) Compute Var(x | C), where C = V \\ (A  union  {x})
        if len(C) == 0:
            # No conditioning yet
            var_C = dist_prior.marginalize_kernel(x, x).squeeze()
        else:
            var_C = self._cond_var_kernel(x, C, dist_prior)

        # 3) Score = log Var ratio
        eps = 1e-12
        var_C = max(var_C, eps)
        return var_A / var_C

    def metric(self, unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False):

        if test_point is None:
            return self._MI_set(selected_set, unselect_set, dist_prior)

        return self._MI_add_point(selected_set, unselect_set, [test_point], dist_prior)

class Min_MI(Mutual_Information):
    def __init__(self, percent_substitution=0.1, *, type_algo = 2):
        super().__init__(type_algo = type_algo, percent_substitution = percent_substitution)
    def metric(self, unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False):
        if test_point is not None:
            return super().metric(unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False)

        min_mi = np.inf
        for point in unselect_set:
            test_set = selected_set
            test_unset =  [point]
            mi = super().metric(test_unset, test_set, None, dist_prior, dist_post)
            if mi < min_mi:
                min_mi = mi
        return min_mi
