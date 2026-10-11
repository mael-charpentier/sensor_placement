import numpy as np

from spgprcpd.placement.base import Metric

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False

class KL_Divergence_Exhaustive(Metric):
    def __init__(self, *, type_algo = 0): # 1):
        super().__init__(type_algo = type_algo)

    def get_d_kl(self, selected_set, dist_prior, dist_post):

        k = len(selected_set)

        dist_prior = dist_prior(A = selected_set)
        dist_post = dist_post(A = selected_set)

        mu0 = dist_prior.marginalize_mu()
        mu1 = dist_post.marginalize_mu()
        Sigma0 = dist_prior.marginalize_kernel()
        Sigma1 = dist_post.marginalize_kernel()

        diff = mu1 - mu0

        logdet0 = np.linalg.slogdet(Sigma0)[1]
        logdet1 = np.linalg.slogdet(Sigma1)[1]

        term_logdet = logdet0 - logdet1
        term_quad = diff.T @ np.linalg.solve(Sigma0, diff)
        term_trace = np.trace(np.linalg.solve(Sigma0, Sigma1))

        return 0.5 * (term_logdet - k + term_quad + term_trace)

    def metric(self, unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False):
        if test_point is not None:
            return self.get_d_kl(list(selected_set) + [test_point], dist_prior, dist_post) - self.get_d_kl(selected_set, dist_prior, dist_post)

        return self.get_d_kl(selected_set, dist_prior, dist_post)

class KL_Divergence(KL_Divergence_Exhaustive):
    """Mean-shift kl criterion as the mixed-integer program."""
    def __init__(self, *, type_algo = 3, big_m=None, big_m_row=None):
        super().__init__(type_algo = type_algo)
        self.big_m = big_m          # bound on |v_i| (support constraint)
        self.big_m_row = big_m_row  # bound on the relaxed rows of Sigma_A w_A = delta_A

    def metric(self, model, num_to_choose, possibilities, cant_choose, dist_prior, dist_post):
        n = len(possibilities)           # |V|
        b = num_to_choose                # cardinality budget

        # mean shift delta = mu^1 - mu^0 and the pre-change covariance Sigma^0 (full)
        delta = np.asarray(dist_post.marginalize_mu() - dist_prior.marginalize_mu(), float)
        Sigma = np.asarray(dist_prior.marginalize_kernel(), float)

        # Two big-M constants, one per constraint family.
        delta_b = min(float(np.linalg.norm(delta)), float(b) ** 0.5 * float(np.abs(delta).max()))

        lam_min = max(float(np.linalg.eigvalsh(Sigma).min()), 1e-9)
        sigma2_max = float(np.diag(Sigma).max())

        # M_v >= |v_i|:
        #   ||v_A||_2 <= ||delta_A||_2 / lam_min(Sigma_A) <= delta_b / lam_min(Sigma)
        if self.big_m is not None:
            M_v = float(self.big_m)
        else:
            M_v = delta_b / lam_min

        # M_row >= |(Sigma v)_i|:
        #   |Sigma_{i,A} Sigma_A^{-1} delta_A| <= sqrt(sigma2_max) * delta_b / sqrt(lam_min),
        if self.big_m_row is not None:
            M_row = float(self.big_m_row)
        else:
            M_row = delta_b * (sigma2_max / lam_min) ** 0.5

        z = model.addVars(n, vtype=GRB.BINARY, name="z")
        v = model.addVars(n, lb=-GRB.INFINITY, name="v")

        # objective: max 0.5 sum_i delta_i v_i  (= 0.5 delta_A^T Sigma_A^{-1} delta_A at optimum)
        model.setObjective(0.5 * gp.quicksum(delta[i] * v[i] for i in range(n)), GRB.MAXIMIZE)

        for i in range(n):
            # v supported on the active set: v_i = 0 when z_i = 0
            model.addConstr(v[i] <= M_v * z[i])
            model.addConstr(v[i] >= -M_v * z[i])
            # active-set system Sigma_A w_A = delta_A: row i is enforced only when z_i = 1
            row = gp.quicksum(Sigma[i, j] * v[j] for j in range(n))
            model.addConstr(row <= delta[i] * z[i] + M_row * (1 - z[i]))
            model.addConstr(row >= delta[i] * z[i] - M_row * (1 - z[i]))

        # cardinality the placement size <= b
        model.addConstr(gp.quicksum(z[i] for i in range(n)) <= b, name="cardinality")

        return z



