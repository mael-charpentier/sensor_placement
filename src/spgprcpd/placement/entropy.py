import numpy as np

from spgprcpd.placement.base import Metric


class Entropy(Metric):
    def __init__(self, *, percent_substitution=0, type_algo=0): # 1):
        super().__init__(type_algo=type_algo, percent_substitution=percent_substitution)

    def _entropy(self, selected_set, dist_prior):
        if len(selected_set) == 0:
            return 0.0  # entropy of empty set is 0

        A = np.array(selected_set)
        Sigma_AA = dist_prior.marginalize_kernel(A, A)
        _, logdet_AA = np.linalg.slogdet(Sigma_AA)

        # entropy of selected set (maximize)
        return 0.5 * (len(A) * np.log(2 * np.pi * np.e) + logdet_AA)

    def metric(self, unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False):
        if test_point is not None:
            return self._entropy(list(selected_set) + [test_point], dist_prior)

        return self._entropy(selected_set, dist_prior)

class Max_Entropy(Entropy):
    """Min-max conditional entropy: minimise the worst unsensed location's uncertainty.

    The target is min_A max_x H(x | A), place sensors so that no location is left
    badly predicted, but the optimisers here maximise, so the objective is written as
    max_A min_x -H(x | A).

    Because this runs as a local search, the first pass optimises H(B | A), which is
    equivalent to optimising -H(A).
    """
    def __init__(self, percent_substitution=0.1, *, type_algo=2):
        super().__init__(type_algo = type_algo, percent_substitution=percent_substitution)

    def __entropy_condi(self, unselect_set, selected_set, dist_prior):
        B = np.array(unselect_set)

        Sigma_BB = dist_prior.marginalize_kernel(B, B)

        if len(selected_set) == 0:
            Sigma_cond_AB = Sigma_BB                 # H(X_B): nothing to condition on yet
        else:
            A = np.array(selected_set)
            Sigma_AA = dist_prior.marginalize_kernel(A, A)
            Sigma_AB = dist_prior.marginalize_kernel(A, B)
            Sigma_cond_AB = Sigma_BB - Sigma_AB.T @ np.linalg.solve(Sigma_AA, Sigma_AB) # sigma of B|A

        _, det_BA = np.linalg.slogdet(Sigma_cond_AB) # det of B|A

        # normaly we want to minimize entropy, but the algo I use will maximize
        # so we will maximize the -entropy
        return - 0.5 * (len(unselect_set) * np.log(2*np.pi*np.e) + det_BA)

    def metric(self, unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False):
        if test_point is not None:
            # forward the flag rather than hardcoding it. It makes no difference today,
            # because the parent ignores second_pass and the only caller that passes it
            # as true also passes test_point as None, so this branch is not reached.
            # Weighted_Metric does forward it to its children, so hardcoding false here
            # would be the one place that silently drops it.
            return super().metric(unselect_set, selected_set, test_point, dist_prior,
                                  dist_post, second_pass=second_pass)

        # worst-case residual uncertainty: -max_{u in V\A} H(X_u | X_A) = min_u (-H(X_u | X_A))
        min_H = np.inf
        for point in unselect_set:
            H = self.__entropy_condi([point], selected_set, dist_prior)
            if min_H > H:
                min_H = H
        return min_H
