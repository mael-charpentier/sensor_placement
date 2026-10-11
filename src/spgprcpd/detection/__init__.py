"""Sequential change point detectors, and the calibration of their thresholds.

Every detector subclasses CPD from the base module and splits its work in two, which is
what makes calibration cheap. The _values method computes a statistic that does not depend
on the threshold, and _alarms applies the threshold to it. Calibration can therefore
compute each statistic once and only re threshold it, instead of rerunning the detector for
every candidate threshold.

Modules:

    base         the CPD base class, the evaluation metrics, and the calibration entry
                 points
    cusum        the likelihood ratio cusum, which needs the post change law, and the
                 normalised cusum, which does not
    ewma         the exponentially weighted moving average, univariate and multivariate,
                 plus a hybrid with bocpd
    bocpd        bayesian online change point detection
    segcpgp      a gaussian process change point kernel with a likelihood ratio test
    ensemble     one univariate detector per sensor, with a vote, which is what lets a
                 record with per sensor gaps be monitored at all
    calibration  bisection on the pre change law, and a grid search when both laws are known
"""
