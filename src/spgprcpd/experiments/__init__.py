"""Benchmark drivers.

Both drivers enumerate a grid of experiment cells, run one cell per invocation so that a
cluster job array can fan them out by index, and write one json per cell into results/.
A cell whose file already exists is skipped, which is what makes a sweep resumable.

Modules:
spgprcpd.experiments.synthetic  synthetic study, where f0 and f1 are both known, so the
                                f1-dependent criteria are available too.
spgprcpd.experiments.real       real-data engine, f0 only, shared by both configs.
spgprcpd.experiments.configs    per-dataset configuration for the real benchmarks.
spgprcpd.experiments.grids      the synthetic sweep definition and the parameter schema.
"""
