"""Benchmark on the real Intel Berkeley Lab record.

Grid = objective x budget x detector x arl0 (see GRID below). Every placement objective is
scored on detection (delay, false alarm rate, F1. Detector calibrated on the real historical f0 to a
common arl0) and on inference (gaussian process reconstruction at the held-out test motes, masked to
actually-observed entries).

This module is declarative: everything below is configuration.  To change what is run,
edit these constants, no driver code changes.
"""
from spgprcpd.data.labdata import Labdata_Data_Manager
from spgprcpd.experiments import real as real_experiment

# data
FIELD = "temperature"             # "temperature" | "humidity" | "light" | "voltage"
SENSORS = None                    # e.g. [1, 2, 3,.] mote ids. None -> best covered
EXCLUDE_SENSORS = (5, 15)         # mote ids to always drop
TIME_RANGE = ("2004-03-01", "2004-03-10")   # dense stretch of the log. None -> full record
RESAMPLE = "15min"                # bin width of the common grid
MIN_COVERAGE = 0.5                # drop motes observed in fewer bins than this fraction
# num_points / N_test values to benchmark (every (count, count, seed) combination = one
# experiment setting, swept as the first grid dimension).  The counts are fixed (the
# lists below). Only which motes fill the roles is random, picked uniformly among those
# passing MIN_COVERAGE, reproducibly from the selection seed.  Several seeds = several
# independent location draws for the same counts (within one seed the draws are nested:
# the 6-mote set extends the 5-mote set by one location).
NUM_POINTS_LIST = [20]          # candidate counts (placement set V) to try
N_TEST_LIST = [20]                 # held-out inference-target counts to try
SELECTION_SEEDS = [0]       # one independent random location draw per seed
SETTINGS = [(n, t, s) for n in NUM_POINTS_LIST for t in N_TEST_LIST
            for s in SELECTION_SEEDS]
MAX_SAMPLES = None                # cap T by even subsampling (long records)
# f0-only phases (real data never exposes f1): SPLIT_TIME = historical/test boundary
# (index or timestamp. None -> T/2). CHANGE_TIME = a known/annotated real event in the
# test stream, or None (default) when no change is assumed, the detectors are then
# scored on staying silent (no_alarm / num_alarms / false alarm rate).
SPLIT_TIME = 100
CHANGE_TIME = None

# experiment grid (setting x objective x budget x detector x arl0)
OBJECTIVES = list(real_experiment.default_objectives())   # registry names
BUDGETS = [3, 5, 7, 10]               # sensors to place among the candidates
DETECTORS = ["cusum_no_post", "ewma_a0.1", "ewma_a0.2"]
TARGET_ARLS = [150.0]             # common false-alarm budget alpha
GRID = [SETTINGS, OBJECTIVES, BUDGETS, DETECTORS, TARGET_ARLS]

SEEDS_CODE = -3                   # seed convention: -N -> seeds 0.N-1
CALIB_STREAMS = 1                # bootstrap windows for the f0 calibration
DO_INFERENCE = True               # gaussian process reconstruction at the held-out motes (needs gpflow)
EXTRAPOLATION = "streaming_sparse"  # "streaming_sparse" | "exact" | "exact_no_retrain"
# Model used to estimate f0 for the placement criteria, separate
# from EXTRAPOLATION above, which performs the online inference afterwards.  Must be an
# exact gaussian process (the sparse / streaming variants do not keep their training data).  None ->
# fall back on the entrywise empirical covariance. See dist/data_real_base.py.
EXTRAPOLATION_EST = "exact"
TAG = "labdata_" + EXTRAPOLATION + "_" + FIELD + "_" + RESAMPLE

PIPELINE_KWARGS = dict(calib_streams=CALIB_STREAMS, do_inference=DO_INFERENCE,
                       extrapolation_name=EXTRAPOLATION)

_dms = {}


def build_dm(setting=None):
    """Memoized data manager per (num_points, N_test, selection_seed) setting (the
    2.3M-row log takes a while to load).  The counts are given. Only which motes fill the
    roles is random, reproducibly drawn with the setting's seed."""
    num_points, n_test, sel_seed = SETTINGS[0] if setting is None else setting
    key = (num_points, n_test, sel_seed)
    if key not in _dms:
        print(f"Loading Intel Berkeley Lab data (setting: num_points={num_points}, "
              f"N_test={n_test}, selection_seed={sel_seed}, {FIELD}, "
              f"resample={RESAMPLE}, time_range={TIME_RANGE}) ...")
        est_cls, est_kernel = real_experiment.estimation_model(EXTRAPOLATION_EST)
        _dms[key] = Labdata_Data_Manager(
            field=FIELD, sensors=SENSORS, num_points=num_points,
            exclude_sensors=EXCLUDE_SENSORS, random_selection=True,
            N_test=n_test, time_range=TIME_RANGE,
            resample=RESAMPLE, min_coverage=MIN_COVERAGE,
            split_time=SPLIT_TIME, change_time=CHANGE_TIME,
            max_samples=MAX_SAMPLES, seed=sel_seed,
            extrapolation_cls=est_cls, kernel_prior_init=est_kernel)
        print(_dms[key].describe())
    return _dms[key]
