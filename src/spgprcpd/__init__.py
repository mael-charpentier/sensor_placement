"""spgprcpd, sensor placement for gaussian process regression and change point detection.

The name is an acronym: sensor placement, gpr, cpd.

Given a set of candidate sensor locations and a budget, the pipeline picks which locations to
instrument, then scores that choice on the two jobs the network has to do:

  * inference, reconstruct the field at the locations that were left unsensed, using a
    gaussian process. See spgprcpd.inference.
  * detection, watch the placed sensors for a change in the distribution of what they
    measure, using a sequential detector. See spgprcpd.detection.

The two pull in different directions, which is the whole point of the comparison.

Subpackages:
    spgprcpd.data         the synthetic generator and the two real records, all behind one
                          data manager interface
    spgprcpd.placement    the placement criteria, meaning entropy, mutual information, kl
                          divergence and the weighted family between them
    spgprcpd.detection    the cusum, ewma, bocpd and segcpgp detectors, plus the
                          data-driven threshold calibration
    spgprcpd.inference    the gaussian process models, exact and streaming sparse
    spgprcpd.experiments  the drivers that sweep a grid of settings and write results
"""

__version__ = "0.1.0"
