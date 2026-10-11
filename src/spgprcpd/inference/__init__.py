"""Gaussian process models that reconstruct the field at the unsensed locations.

Every model subclasses spgprcpd.inference.base.Extrapolation and shares the same streaming
contract, which is update(X, y) and then predict(X_test), called once per batch. eval scores
the predictions accumulated that way and reports the mean squared error, the 95% coverage,
how many points were scored, the training likelihood and the recovered kernel parameters.

Modules:
    base     the Extrapolation base class, the jitter retry, and the metrics
    exact    the exact gaussian process ladder, three variants that differ only in how often
             the kernel is fitted, which are Exact_GP on every batch,
             Exact_GP_no_retraining on the first batch only, and Exact_GP_no_training never,
             the last one being handed the true kernel
    sparse   streaming sparse gaussian process with inducing points
    pep      the power expectation propagation backend that sparse is built on
"""
