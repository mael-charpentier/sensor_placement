"""Data managers and the probability model they are built on.

Every manager exposes the same interface, so the rest of the pipeline runs unchanged on
synthetic data and on a real record:

    dm.V                   candidate locations, one row per location
    dm.dist_prior          the pre-change law
    dm.dist_post           the post-change law, or None when the data has no such law
    dm.get_dataset(A, B)   the training and test tensors for a placement
    dm.change_time         when the change happens, or None

Modules:
    gaussian    the multivariate gaussian over space and time that the criteria and the
                detectors consume, plus the helpers that read one off a fitted gaussian
                process
    kernels     a registry of named kernels, indexed by name, time handling and variant
    synthetic   a manager that samples from a known pre-change and post-change law
    real_base   the shared base for the two real records
    labdata     the Intel Berkeley Lab record
    ecosense    the EcoSense Montreal deployment
"""
