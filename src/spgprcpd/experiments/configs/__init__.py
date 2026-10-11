"""Per-dataset configuration for the real-data benchmarks.

Each module is declarative: module-level constants describing the record (field, sensor
roles, time range, resampling, phase split) and the experiment grid, plus a memoized
build_dm returning the configured data manager.  Editing an experiment means editing
these constants, no driver code changes.  spgprcpd.experiments.real consumes them.
"""
