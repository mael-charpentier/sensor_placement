import itertools

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False


class Metric:
    def __init__(self, type_algo, percent_substitution=0.5):
        """type_algo picks the optimiser: 0 greedy, 1 exhaustive, 2 hill climbing local
        search started from greedy, 3 the mixed integer program solved with gurobi.

        percent_substitution is only read by the local search, see local for how it is
        turned into a swap budget.
        """
        self.type_algo = type_algo
        self.percent_substitution = percent_substitution
        self.normalize = False


    def metric(self, selected_set, test_point, dist_prior, dist_post, second_pass = False):
        raise NotImplementedError

    def exhaustive(self, num_to_choose, possibilities, cant_choose, dist_prior, dist_post):
        best_subset = []
        best_val = -np.inf

        second_pass=False

        if self.normalize:
            for subset in itertools.combinations(possibilities, num_to_choose):

                subset = list(subset)
                unsubset = [x for x in possibilities+cant_choose if x not in subset]

                self.metric(unsubset, subset, None, dist_prior, dist_post, second_pass=second_pass)
            second_pass = True

        for subset in itertools.combinations(possibilities, num_to_choose):

            subset = list(subset)
            unsubset = [x for x in possibilities+cant_choose if x not in subset]

            val = self.metric(unsubset, subset, None, dist_prior, dist_post, second_pass=second_pass)

            if val > best_val:
                best_val = val
                best_subset = [subset]
            elif val == best_val:
                best_subset.append(subset)

        # ties are collected above, we just return the first one
        return best_subset[0], best_val

    def greedy(self, num_to_choose, possibilities, cant_choose, dist_prior, dist_post):
        A = []

        for _ in range(num_to_choose):

            best = None
            best_val = -np.inf

            for v in possibilities:

                new_unset = list(set(possibilities+cant_choose) - {v})

                val = self.metric(new_unset, A, v, dist_prior, dist_post)
                if val > best_val:
                    best_val = val
                    best = v

            A.append(best)
            possibilities.remove(best)

        # score the placement that was built, rather than returning best_val. The loop
        # above compares marginal gains, so best_val is the gain of the last pick.
        return A, self.metric(possibilities + cant_choose, A, None, dist_prior, dist_post)

    def local(self, num_to_choose, possibilities, cant_choose, dist_prior, dist_post):
        """Search around the greedy placement by swapping sensors in and out.

        percent_substitution is turned into a swap budget k:

            value <= 1   k = ceil(budget * value), so a fraction of the budget
            value > 1    k = floor(value), so an absolute count

        If k reaches the budget, the method falls back to exhaustive, a full
        enumeration of the placements.

        This is the hill climbing local search. It repeats "take the single best improving
        swap, then re anchor on the result" and stops as soon as no single swap improves,
        and in any case after k of those steps.

        A different algorithm would be to take the greedy placement as a fixed anchor,
        then for n = 1 up to k, try every way of removing n sensors from the anchor and
        adding n from the unselected locations, and keep the best placement seen. The
        anchor never moves and the loop never stops early.
        """
        if self.percent_substitution <= 1:
            # read as a fraction of the budget
            num_substitution = int(np.ceil(num_to_choose * self.percent_substitution))
        else:
            # read as an absolute number of swaps
            num_substitution = int(np.floor(self.percent_substitution))
        if num_substitution >= num_to_choose:
            return self.exhaustive(num_to_choose, possibilities, cant_choose, dist_prior, dist_post)


        subset, val_greedy = self.greedy(num_to_choose, possibilities, cant_choose, dist_prior, dist_post)
        if num_substitution == 0:
            return subset, val_greedy

        unselected = [x for x in possibilities if x not in subset]
        unselected_total = [x for x in possibilities+cant_choose if x not in subset]

        best_subset = subset.copy()
        best_val = val_greedy

        for _ in range(num_substitution):
            best_swap = None
            step_val = best_val

            # every way of taking one sensor out and one unselected location in. Only the
            # locations in possibilities can come in, so anything in cant_choose stays in
            # unselected_total and is never a candidate
            for old_val in best_subset:
                for new_val in unselected:
                    new_subset = [x for x in best_subset if x != old_val] + [new_val]
                    new_unset = [x for x in unselected_total if x != new_val] + [old_val]

                    val = self.metric(new_unset, new_subset, None, dist_prior, dist_post)
                    if val > step_val:
                        step_val = val
                        best_swap = (old_val, new_val)

            if best_swap is None:
                # no single swap improves on where we are, so the placement is a local
                # optimum for this neighbourhood and a further step would find nothing.
                # Requiring a strict improvement is also what keeps the loop from cycling
                break

            # re anchor on the winner: the sensor that moved out becomes a candidate again,
            # the one that moved in stops being one
            old_val, new_val = best_swap
            best_subset = [x for x in best_subset if x != old_val] + [new_val]
            unselected = [x for x in unselected if x != new_val] + [old_val]
            unselected_total = [x for x in unselected_total if x != new_val] + [old_val]
            best_val = step_val

        return best_subset, best_val

    def gurobi(self, num_to_choose, possibilities, cant_choose, dist_prior, dist_post):
        if not _GUROBI_AVAILABLE:
            raise ImportError("gurobipy is required for the Gurobi optimizer.")

        model = gp.Model("KL_subset_selection")
        x = self.metric(model, num_to_choose, possibilities, cant_choose, dist_prior, dist_post)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            A = [i for i in range(len(possibilities)) if x[i].x > 0.5]
            return A, self.get_d_kl(A, dist_prior, dist_post)
        else:
            raise ValueError("Model not optimal")

    def __call__(self, num_to_choose, possibilities, cant_choose, dist_prior, dist_post):
        if self.type_algo == 0:
            return self.greedy(num_to_choose, possibilities, cant_choose, dist_prior, dist_post)
        elif self.type_algo == 1:
            return self.exhaustive(num_to_choose, possibilities, cant_choose, dist_prior, dist_post)
        elif self.type_algo == 2:
            return self.local(num_to_choose, possibilities, cant_choose, dist_prior, dist_post)
        elif self.type_algo == 3:
            return self.gurobi(num_to_choose, possibilities, cant_choose, dist_prior, dist_post)

    def __add__(self, other):
        if other == 0:
            return self
        return Weighted_Metric(self, other, lambda x,y: x+y)
    def __sub__(self, other):
        if other == 0:
            return self
        return Weighted_Metric(self, other, lambda x,y: x-y)
    def __mul__(self, other):
        if other == 0:
            return 0
        if other == 1:
            return self
        return Weighted_Metric(self, other, lambda x,y: x*y)
    def __truediv__(self, other):
        if other == 1:
            return self
        return Weighted_Metric(self, other, lambda x,y: x/y)
    def __radd__(self, other):
        if other == 0:
            return self
        return Weighted_Metric(other, self, lambda x,y: x+y)
    def __rsub__(self, other):
        return Weighted_Metric(other, self, lambda x,y: x-y)
    def __rmul__(self, other):
        if other == 0:
            return 0
        if other == 1:
            return self
        return Weighted_Metric(other, self, lambda x,y: x*y)
    def __rtruediv__(self, other):
        if other == 0:
            return 0
        return Weighted_Metric(other, self, lambda x,y: x/y)

class Weighted_Metric(Metric):
    def __init__(self, metric_1, metric_2, op, *, percent_substitution=0, type_algo = 0, normalize=True, norm_max = 100):
        def get_algo(algo_1, algo_2): #  -> not exactly true, because depend on the op and metric, but for the report it is enough
            if algo_1 == 3 or algo_2 == 3: # if one of the 2 is optimize, then can't do
                raise ValueError("Can't optimize a weighted metric, when it use gurobi.")
            if algo_1 == 1 or algo_2 == 1: # if one of the 2 is exhaustive
                return 1
            elif algo_1 == 2 or algo_2 == 2: # if one of the 2 is local search
                return 2
            return 0 # if both are greedy

        if isinstance(metric_1, Metric) and isinstance(metric_2, Metric):
            type_algo = get_algo(metric_2.type_algo, metric_1.type_algo)
        elif isinstance(metric_1, Metric):
            type_algo = metric_1.type_algo
        elif isinstance(metric_2, Metric):
            type_algo = metric_2.type_algo

        super().__init__(type_algo=type_algo, percent_substitution=percent_substitution)
        self.metric_1 = metric_1
        self.metric_2 = metric_2
        self.op = op
        self.normalize = normalize
        self.norm_max = norm_max
        self.norm_cache_1 = {}
        self.norm_cache_2 = {}
        self.min_1, self.max_1 = None, None
        self.min_2, self.max_2 = None, None

    def metric(self, unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = False):
        t_subset = tuple(selected_set)
        if self.normalize and second_pass:
            if self.min_1 is None:
                if self.norm_cache_1 != {}:
                    self.min_1 = min(self.norm_cache_1.values())
                    self.max_1 = max(self.norm_cache_1.values())
                else:
                    self.min_1 = 0
                    self.max_1 = 1 - 1e-12

                if self.norm_cache_2 != {}:
                    self.min_2 = min(self.norm_cache_2.values())
                    self.max_2 = max(self.norm_cache_2.values())
                else:
                    self.min_2 = 0
                    self.max_2 = 1 - 1e-12

        if isinstance(self.metric_1, Metric):
            if self.normalize and second_pass and not isinstance(self.metric_2, Weighted_Metric):
                val_1 = self.norm_max * ((self.norm_cache_1[t_subset] - self.min_1) / (self.max_1 - self.min_1 + 1e-12))
            else:
                val_1 = self.metric_1.metric(unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = second_pass)
                if self.normalize and not isinstance(self.metric_2, Weighted_Metric):
                    self.norm_cache_1[t_subset] = val_1
        else:
            val_1 = self.metric_1

        if isinstance(self.metric_2, Metric):
            if self.normalize and second_pass and not isinstance(self.metric_2, Weighted_Metric):
                val_2 = self.norm_max * ((self.norm_cache_2[t_subset] - self.min_2) / (self.max_2 - self.min_2 + 1e-12))
            else:
                val_2 = self.metric_2.metric(unselect_set, selected_set, test_point, dist_prior, dist_post, second_pass = second_pass)
                if self.normalize and not isinstance(self.metric_2, Weighted_Metric):
                    self.norm_cache_2[t_subset] = val_2
        else:
            val_2 = self.metric_2

        return self.op(val_1, val_2)



