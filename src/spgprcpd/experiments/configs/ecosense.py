"""Benchmark on the real EcoSense Montreal deployment.

Grid = objective x budget x detector x arl0 (see GRID below). Every placement objective is
scored on detection (delay, false alarm rate, F1. Detector calibrated on the real historical f0 to a
common arl0) and on inference (gaussian process reconstruction at the evaluation sensors, masked to
actually-observed entries).

This module is declarative: everything below is configuration.  To change what is run, edit
these constants, no driver code changes.
"""
from spgprcpd.data.ecosense import Ecosense_Data_Manager
from spgprcpd.experiments import real as real_experiment

# data
SOURCE = "all"                    # key of spgprcpd.data.ecosense.SOURCES, or a csv path
FIELD = "PM2.5 (ug/m3)"           # channel, see Ecosense_Data_Manager for what else is in the log
# sensor roles (a sensor may appear in several lists):
PLACEMENT_SENSORS = [9, 17, 18, 19]    # can be selected by the placement criteria
EVALUATION_SENSORS = [7, 9, 17, 18, 19]           # inference evaluated there (18: only when not placed).                                       # masked to actually-observed entries
HISTORICAL_SENSORS = [9, 17, 18, 19, 20]   # feed the empirical f0/f1 estimation
TIME_RANGE = ("2025-06-05", "2025-07-15 12:00")
RESAMPLE = "30min"                # bin width of the common grid
MIN_COVERAGE = 0.3                # auto-selection threshold. Listed sensors only warn
DROP_FLAT_RUN = 8                 # nan-out >= this many consecutive identical bins
                                  # (dead/stuck sensor flat lines, e.g. Sensor 20's tail)
NUM_POINTS = None                 # placement cap when PLACEMENT_SENSORS is None
# f0-only phases (real data never exposes f1): SPLIT_TIME = historical/test boundary
# (index or timestamp. None -> T/2). CHANGE_TIME = a known/annotated real event in the
# test stream, or None when no change is assumed, the detectors are then scored on
# staying silent (no_alarm / num_alarms / false alarm rate).
# The annotated change is the 2025 Canadian Grand Prix at Parc Jean-Drapeau (report
#): race weekend June 13-15, 2025 (practice starts Friday the 13th).
# Historical f0 = June 6-10 (normal regime), test stream starts June 10 and contains
# the event.
SPLIT_TIME = "2025-06-10"
CHANGE_TIME = "2025-06-13"

# experiment grid (objective x budget x detector x arl0)
OBJECTIVES = list(real_experiment.default_objectives())   # registry names
BUDGETS = [2, 3]               # sensors to place among the candidates
# The EcoSense record keeps its per-sensor gaps (they are not interpolated), so only the
# per-location ensembles apply here: each sub-detector skips its own nan.  A joint
# multivariate detector would see nan in the observation vector, see real.default_detectors.
DETECTORS = ["cusum_no_post_loc", "ewma_a0.1_loc",
             "cusum_no_post_loc_maj", "ewma_a0.1_loc_maj"]
TARGET_ARLS = [100.0]             # common false-alarm budget alpha (short record -> modest)
GRID = [OBJECTIVES, BUDGETS, DETECTORS, TARGET_ARLS]

SEEDS_CODE = -3                   # seed convention: -N -> seeds 0.N-1
CALIB_STREAMS = 1                 # bootstrap windows for the f0 calibration
DO_INFERENCE = True               # gaussian process reconstruction at the evaluation sensors (gpflow)
EXTRAPOLATION = "streaming_sparse"  # "streaming_sparse" | "exact" | "exact_no_retrain"
# Model used to estimate f0 for the placement criteria, separate
# from EXTRAPOLATION above, which performs the online inference afterwards.  It must be an
# exact gaussian process: reading the posterior mean off a fitted model needs the full training set,
# which the sparse / streaming variants do not keep.  Set to None to fall back on the
# entrywise empirical covariance, but note that the fallback cannot rank a sensor that is
# absent from the historical record (e.g. EcoSense sensor 7), see data_real_base.
EXTRAPOLATION_EST = "exact"
TAG = "ecosense_" + EXTRAPOLATION + "_" + FIELD + "_" + RESAMPLE

PIPELINE_KWARGS = dict(calib_streams=CALIB_STREAMS, do_inference=DO_INFERENCE,
                       extrapolation_name=EXTRAPOLATION)

_dms = {}


def build_dm(_field=FIELD):
    """Memoized data manager (the full log takes ~15 s to load).

    Keyed on FIELD rather than held in a single slot, because callers switch channel by
    reassigning the module-level FIELD (a sweep running temperature and PM2.5 in one
    process does this).  A single-slot cache would hand the second channel the first
    channel's manager: the run would then be tagged as one channel while being scored on
    another, and two channels reporting identical numbers is the symptom."""
    if _field not in _dms:
        print(f"Loading EcoSense ({SOURCE}, {_field}, resample={RESAMPLE}) ...")
        est_cls, est_kernel = real_experiment.estimation_model(EXTRAPOLATION_EST)
        _dms[_field] = Ecosense_Data_Manager(field=_field, source=SOURCE,
                                    extrapolation_cls=est_cls,
                                    kernel_prior_init=est_kernel,
                                    placement_sensors=PLACEMENT_SENSORS,
                                    evaluation_sensors=EVALUATION_SENSORS,
                                    historical_sensors=HISTORICAL_SENSORS,
                                    num_points=NUM_POINTS,
                                    time_range=TIME_RANGE, resample=RESAMPLE,
                                    min_coverage=MIN_COVERAGE,
                                    drop_flat_run=DROP_FLAT_RUN,
                                    split_time=SPLIT_TIME,
                                    change_time=CHANGE_TIME)
        print(_dms[_field].describe())
    return _dms[_field]
