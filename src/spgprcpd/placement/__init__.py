"""Sensor-placement criteria.

Every criterion subclasses spgprcpd.placement.base.Metric and is called as

    A, value = criterion(budget, candidates, cannot_choose, dist_prior, dist_post)

Metric supplies the optimisers (greedy, exhaustive, local search after greedy), so a
criterion only has to implement metric(), which returns the score of a placement, or, when
test_point is given, the marginal gain of adding that point.  Criteria compose
arithmetically, as in 0.5 * Mutual_Information() + 0.5 * KL_Divergence_Exhaustive().

Modules:
spgprcpd.placement.base               Metric base class and the optimisers.
spgprcpd.placement.entropy            max-entropy and the min-max conditional-entropy variant.
spgprcpd.placement.mutual_information mutual information and its min variant.
spgprcpd.placement.kl_divergence      kl between f0 and f1 (detection-oriented).
"""
