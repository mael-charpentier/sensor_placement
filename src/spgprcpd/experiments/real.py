"""Shared experiment engine for the real-data benchmarks, driven by configs.ecosense and
configs.labdata. It is the real-data counterpart of experiments/synthetic.py.

A real historical record only ever shows the normal regime f0. There is no second regime to
read off it, so the real data managers return dist_post=None and every stage falls back to
its f0-only variant:

  1. placement. The objective picks budget sensors out of the candidates using the empirical
     dist_prior. The criteria that need f0 alone are entropy, max_H, MI and min_MI. The ones
     that also need f1 sit behind default_objectives(with_post=True), so they only show up
     on synthetic-style runs.
  2. detection. Only the f0-only detectors are used, meaning the normalised cusum and the
     ewma family. The likelihood-ratio cusum is excluded because its statistic needs f1.
     Each threshold is calibrated on the real historical f0 by moving-block bootstrap,
     aiming at the arl0 that every detector shares, then the detector runs on the held-out
     test stream. When the manager has no annotated change, so test_change_time is None, the
     stream is taken to be change-free and the detector is scored on staying quiet, which is
     what cpd.eval(None) reports as no_alarm, num_alarms and false_alarm_rate.
  3. inference. A gaussian process is fitted on the record of the placed sensors over the
     test phase and scored at the evaluation sensors by Extrapolation.eval, which returns
     the mean squared error and the 95% coverage. Entries that were never observed are
     masked out. This stage needs gpflow and is skipped with a warning when it is missing.

"""
import csv
import itertools
import json
import os
import time

import numpy as np

from spgprcpd.detection import calibration as cal
from spgprcpd.detection import cusum, ensemble, ewma
from spgprcpd.placement import entropy, kl_divergence, mutual_information


# registries (subset / override from the benchmark configs)
def estimation_model(name="exact", kernel="Matern_Anisotropic"):
    """(extrapolation_cls, fresh init kernel) used to estimate f0 for the placement
    criteria, i.e. The estimation stage.  The data managers take
    these and fit the kernel to the historical record, instead of
    estimating the covariance entry by entry (see spgprcpd.data.real_base.kernel_gaussian for
    why that distinction decides which sensors can be ranked at all).

    name must be an exact gaussian process: reading the posterior mean off the fitted model needs the
    full training set, which the sparse / streaming variants do not keep.  This is separate
    from the model that performs the online inference afterwards.

    Returns (None, None), so the caller falls back on the empirical covariance, when
    gpflow is missing or no model is requested, matching how the rest of this module
    degrades without gpflow.
    """
    if name is None:
        return None, None
    try:
        from spgprcpd.data import kernels
        from spgprcpd.inference import exact
    except ImportError as e:
        print(f"[warn] estimation_model: no gpflow ({e}); f^0 will fall back to the "
              f"empirical covariance")
        return None, None
    # any exact variant is eligible: they all keep their full training set, which is what
    # copy_posterior_mean needs (the sparse / streaming models do not)
    register = {"exact": exact.Exact_GP,
                "exact_no_retrain": exact.Exact_GP_no_retraining,
                "exact_no_train": exact.Exact_GP_no_training}
    if name not in register:
        raise ValueError(f"unknown estimation model {name!r}; choose from {list(register)} "
                         f"(it must keep its training data)")
    return register[name], kernels.kernels[kernel]["time"]["init_learning"]()


def default_objectives(with_post=False):
    """name -> factory of a fresh placement criterion (fresh: Weighted_Metric and the local
    searches keep per-run caches).  Default = the f0-only criteria: real data has no f1
    (dist_post is None), so the kl-based criteria only appear with with_post=True
    (synthetic-style runs where a post-change distribution exists)."""
    objectives = {
        "entropy": lambda: entropy.Entropy(),
        "max_H":   lambda: entropy.Max_Entropy(),
        "MI":      lambda: mutual_information.Mutual_Information(),
        "min_MI":  lambda: mutual_information.Min_MI(),
    }
    if with_post:
        objectives.update({
            "KL":    lambda: kl_divergence.KL_Divergence_Exhaustive(),
            "MI+KL": lambda: 0.5 * mutual_information.Mutual_Information()
                             + 0.5 * kl_divergence.KL_Divergence_Exhaustive(),
        })
    return objectives


def per_location(sub_class_factory, mode="any"):
    """Wrap a base detector into the per-location ensemble, meaning one univariate detector
    per sensor, each skipping its own nan steps, and a system alarm when mode of them fire,
    where mode is "any" or "majority" or an int.  Required on the EcoSense record, whose
    per-sensor gaps are kept rather than interpolated, so a multivariate detector would see
    nan in the joint vector.  sub_class_factory(p, q, **kw) builds one base detector with kw
    passed through, that being threshold and alpha_arl."""
    return lambda dp, dq, arl: ensemble.Multiple_CPD(sub_class_factory, dp, dq,
                                                     mode=mode, alpha_arl=arl)


def default_detectors():
    """name -> factory(dist_prior, dist_post, alpha_arl). Thresholds are left undefined so
    they are calibrated on the real f0 (never a hand-picked value).  f0-only detectors
    only: the likelihood-ratio cusum needs the post-change f1, which real
    data never provides.

    The _loc variants are the per-location ensembles. Use them on data with gaps, meaning
    EcoSense, where each sensor skips its own nan. The plain multivariate variants need a
    gap-free joint stream, since they compute a joint Mahalanobis or multivariate ewma
    statistic."""
    _cusum = lambda p, q, **kw: cusum.CuSum_Without_Post(p, q, k=0.5, **kw)
    _ewma1 = lambda p, q, **kw: ewma.EWMA(0.1, p, q, **kw)
    return {
        # multivariate (joint) detectors, need a dense stream
        "cusum_no_post": lambda dp, dq, arl: cusum.CuSum_Without_Post(dp, dq, k=0.5,
                                                                      alpha_arl=arl),
        "ewma_a0.1":     lambda dp, dq, arl: ewma.EWMA(0.1, dp, dq, alpha_arl=arl),
        "ewma_a0.2":     lambda dp, dq, arl: ewma.EWMA(0.2, dp, dq, alpha_arl=arl),
        # per-location ensembles, skip per-sensor nan, alarm if any sensor fires
        "cusum_no_post_loc":     per_location(_cusum, "any"),
        "ewma_a0.1_loc":         per_location(_ewma1, "any"),
        # majority-vote variants (alarm only when most sensors fire)
        "cusum_no_post_loc_maj": per_location(_cusum, "majority"),
        "ewma_a0.1_loc_maj":     per_location(_ewma1, "majority"),
    }


# building blocks
def _from_registry(name, registry, what):
    """Resolve a registry name to its factory, or pass a factory through unchanged.

    Fails immediately, and says what the valid names are, rather than raising a bare
    KeyError from inside the run.  This matters because the sweep driver catches
    exceptions into a log to keep a job array alive: a mistyped name in a config would
    otherwise show up only as a silently missing result file.
    """
    if not isinstance(name, str):
        return name
    if name not in registry:
        raise KeyError(
            f"unknown {what} {name!r}; available: {sorted(registry)}")
    return registry[name]


def validate_grid(general_info):
    """Check every registry name in a grid up front, before any data is loaded.

    general_info is the grid passed to do_exp_general, i.e.
    [objectives, budgets, detectors, target_arls] optionally preceded by settings.
    """
    objectives, detectors = default_objectives(), default_detectors()
    dims = list(general_info)
    if len(dims) == 5:
        dims = dims[1:]                     # drop the leading data-settings dimension
    obj_names, _budgets, det_names, _arls = dims
    for name in obj_names:
        _from_registry(name, objectives, "objective")
    for name in det_names:
        _from_registry(name, detectors, "detector")
    return True


def calibrate_on_record(det, A, B, y_iid_f0, target_arl, n_streams, horizon, rng):
    """Calibrate det.threshold to target_arl on the real in-control record by
    moving-block bootstrap, preserves the real temporal dependence, unlike simulating
    from the fitted Gaussian."""
    streams = cal.bootstrap_streams(y_iid_f0, n_streams, horizon, rng)
    det.threshold = cal.calibrate_f0(det._rebuild_with_threshold, B, A, streams,
                                     target_arl, det._default_bracket(),
                                     expand=det._calib_expand())
    det._deferred_calibration = False
    return det.threshold


def _eval_cols(dm, B):
    """Columns get_dataset will return as test targets, given the manager's
    only_testing_loc convention (for aligning the observation mask)."""
    if dm.only_testing_loc == 1:
        return list(range(dm.num_points, dm.num_points + dm.N_test))
    if dm.only_testing_loc == 2:
        return list(B) + list(range(dm.num_points, dm.num_points + dm.N_test))
    return list(B)


# one experiment cell (the analogue of benchmark.pipeline)
def pipeline_one(dm, objective, budget, detector, *, target_arl=150.0,
                 calib_streams=20, calib_horizon=None, do_inference=True,
                 extrapolation_name="exact", learn_kernel="Matern_Anisotropic",
                 batch_size=50, seeds_code=0):
    """Placement -> calibration -> detection -> inference for one cell. Objective and
    detector are registry names (or factories).  Returns a dict shaped like the
    synthetic spgprcpd.experiments.synthetic.pipeline result."""
    np.random.seed(seeds_code)
    obj_factory = _from_registry(objective, default_objectives(), "objective")
    det_factory = _from_registry(detector, default_detectors(), "detector")
    if calib_horizon is None:
        # streams must be long enough to observe run lengths up to the target arl0
        calib_horizon = max(300, int(2 * target_arl))
    tau = dm.test_change_time            # None -> no change assumed in the test stream
    # everything after the candidates (test + hist-only columns) exists but is not placeable
    cant_choose = list(range(dm.num_points, len(dm.V)))

    # placement
    np.random.seed(seeds_code)
    time_begin = time.time_ns()
    budget = int(min(budget, dm.num_points))
    A, val_A = obj_factory()(budget, list(range(dm.num_points)), list(cant_choose),
                             dm.dist_prior, dm.dist_post)
    A = sorted(int(a) for a in A)
    B = [i for i in range(dm.num_points) if i not in A]
    time_metric = time.time_ns() - time_begin

    # detection: calibrate on the real historical f0, run the test stream
    np.random.seed(seeds_code)
    time_begin = time.time_ns()
    hist_stream = dm.get_dataset(A, B, time_dimension=True, normalize=False,
                                 phase="historical")[0][2]
    y_iid_f0 = hist_stream[1:]                  # the whole historical phase is f0
    y_iid_test = dm.get_dataset(A, B, time_dimension=True, normalize=False,
                                phase="test")[0][2]
    rng = np.random.RandomState(seeds_code)
    det = det_factory(dm.dist_prior, dm.dist_post, target_arl)
    calibrate_on_record(det, A, B, y_iid_f0, target_arl, calib_streams,
                        calib_horizon, rng)
    cpd_param = det.get_all_params()
    time_calibrate_cpd = time.time_ns() - time_begin

    time_begin = time.time_ns()
    det.alarms(B, A, y_iid_test, time_local=True)
    cpd_result = det.eval(tau)
    time_cpd = time.time_ns() - time_begin

    # inference at the evaluation sensors (Extrapolation.eval, masked)
    # placement sensors that must also be evaluated (dm.eval_extra) join the targets
    # whenever they are not placed (only_testing_loc == 2 appends the eval-only columns)
    np.random.seed(seeds_code)
    eval_B = B
    if getattr(dm, "eval_extra", None):
        eval_B = [c for c in dm.eval_extra if c not in A]

    extrapolation_result = None
    time_extra = 0
    if do_inference:
        time_begin = time.time_ns()
        try:
            extrapolation_result = _run_inference(dm, A, eval_B, extrapolation_name,
                                                  learn_kernel, batch_size)
        except ImportError as e:
            print(f"  [warn] inference skipped (missing dependency: {e})")
        except Exception as e:
            print(f"  [warn] inference failed for {objective} k={budget}: {e}")
            extrapolation_result = {"error": str(e)}
        time_extra = time.time_ns() - time_begin

    return {
        "extrapolation": extrapolation_result,
        "cpd": cpd_result,
        "cpd_param": cpd_param,
        "sensor": A,
        "sensor_ids": [dm.sensor_ids[a] for a in A],
        "val": None if val_A is None else float(val_A),
        "time": {
            "metric": time_metric,
            "calibration_cpd": time_calibrate_cpd,
            "cpd": time_cpd,
            "extrapolation": time_extra,
        },
    }


def _run_inference(dm, A, B, extrapolation_name, learn_kernel, batch_size):
    """Train a gaussian process on A's test-phase record, predict the evaluation columns and score with
    Extrapolation.eval, masked to actually-observed entries.  Lazy gpflow imports so
    the CPD part runs without it."""
    from spgprcpd.data import kernels
    from spgprcpd.inference import exact, sparse
    register = {
        "exact": exact.Exact_GP,                              # fit every batch
        "exact_no_retrain": exact.Exact_GP_no_retraining,     # fit first batch only
        "exact_no_train": exact.Exact_GP_no_training,         # never fit (oracle)
        "streaming_sparse": sparse.Streaming_Sparse_GP,
    }
    kern = kernels.kernels[learn_kernel]["time"]["init_learning"]()
    extra = register[extrapolation_name](kern)

    batches = dm.get_dataset(A, B, time_dimension=False, normalize=True,
                             batch_size=batch_size, save_norm=True)
    for X_train, y_train, _, X_test, _, _ in batches:
        if X_train.shape[0] != 0:
            extra.update(X_train, y_train)
        # Predict on every batch, even the ones with no training data: eval pairs
        # _history['means'] positionally with y_test, so a skipped predict drops that
        # batch and shifts all later predictions, corrupting mean squared error/coverage.
        # (experiments/synthetic.py likewise predicts unconditionally.)
        extra.predict(X_test)

    y_test = dm.get_dataset(A, B, time_dimension=False, normalize=True,
                            batch_size=None)[0][4]
    # gap-filled test values are queries for the gaussian process, not ground truth
    mask = None
    if getattr(dm, "observed", None) is not None:
        mask = dm.observed_mask(_eval_cols(dm, B), phase="test")
    return extra.eval(y_test, dm.dist_prior, dm.dist_post, mask=mask)


# job-array driver: one experiment id -> all seeds -> one json
def get_name(tag, objective, budget, detector, target_arl, extrapolation_name, seeds_code):
    name = (f"{tag}_{seeds_code}_{budget}_{target_arl}_{extrapolation_name}"
            f"_{objective}_{detector}.json")
    for ch in " /()":
        name = name.replace(ch, "_" if ch in " /" else "")
    return name


def run_seeds(seeds_code):
    """Seed convention: a negative value -N means the seed set 0.N-1, a
    non-negative value is the single seed itself."""
    return list(range(-seeds_code)) if seeds_code < 0 else [seeds_code]


def do_exp_general(do_id, dm_factory, tag, general_info, *, seeds_code=0,
                   results_dir="results", **pipeline_kwargs):
    os.makedirs(results_dir, exist_ok=True)
    validate_grid(general_info)     # fail on a mistyped registry name before loading data
    exp_id = -1
    for combo in itertools.product(*general_info):
        exp_id += 1
        if do_id != exp_id:
            continue

        if len(combo) == 5:
            setting, objective, budget, detector, target_arl = combo
            set_tag = ("_".join(map(str, setting))
                       if isinstance(setting, (tuple, list)) else str(setting))
            tag_run = f"{tag}_set{set_tag}"
        else:
            setting = None
            objective, budget, detector, target_arl = combo
            tag_run = tag

        result_filename = os.path.join(
            results_dir, get_name(tag_run, objective, budget, detector, target_arl,
                                  pipeline_kwargs.get("extrapolation_name", "exact"),
                                  seeds_code))
        if os.path.exists(result_filename):
            print(f"Already done: {result_filename}")
            return True, None

        print(f"Do this file {result_filename}")
        time_begin = time.time_ns()
        dm = dm_factory() if setting is None else dm_factory(setting)
        all_results = []
        for seed in run_seeds(seeds_code):
            r = pipeline_one(dm, objective, budget, detector, target_arl=target_arl,
                             seeds_code=seed, **pipeline_kwargs)
            if r is not None:
                all_results.append(r)
        time_end = time.time_ns() - time_begin

        if all_results:
            result = {
                "describe": dm.describe(),
                "result": all_results,
                "param": [tag_run, objective, budget, detector, target_arl,
                          dict(pipeline_kwargs), seeds_code, setting],
            }
            with open(result_filename, "w") as _f:
                json.dump(result, _f, indent=4,
                          default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
            print(f"Result saved to {result_filename}, in {time_end * 1e-9:.3f}s")
        else:
            print(f"No result in {result_filename}, in {time_end * 1e-9:.3f}s")
        return True, None

    return False, exp_id


def do_all_exp(dm_factory, tag, general_info, *, seeds_code=0, results_dir="results",
               **pipeline_kwargs):
    """Convenience local driver: run every experiment id of the grid sequentially."""
    exp_id = 0
    while True:
        finish, last_id = do_exp_general(exp_id, dm_factory, tag, general_info,
                                         seeds_code=seeds_code, results_dir=results_dir,
                                         **pipeline_kwargs)
        if not finish:
            print(f"All {last_id + 1} experiments done.")
            return
        exp_id += 1


# single-process summary-table mode (thin wrapper over pipeline_one)
def _mean(xs):
    xs = [v for v in xs
          if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(xs)) if xs else float("nan")


def _fmt(v, w, p=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " " * (w - 3) + "nan"
    return f"{v:{w}.{p}f}"


def run_benchmark(dm, *, name, budgets, objectives=None, detectors=None,
                  target_arl=150.0, calib_streams=20, calib_horizon=None,
                  n_random=5, do_inference=True, extrapolation_name="exact",
                  learn_kernel="Matern_Anisotropic", batch_size=50, seed=0,
                  results_dir="results"):
    objectives = default_objectives() if objectives is None else objectives
    detectors = default_detectors() if detectors is None else detectors
    print(dm.describe())
    print(f"objectives={list(objectives)}  budgets={list(budgets)}  "
          f"detectors={list(detectors)}  ARL0={target_arl:.0f}\n")

    rows = []
    for obj_name, obj_factory in objectives.items():
        for budget in budgets:
            n_rep = n_random if obj_name == "random" else 1
            for rep in range(n_rep):
                for det_name, det_factory in detectors.items():
                    r = pipeline_one(dm, obj_factory, budget, det_factory,
                                     target_arl=target_arl, calib_streams=calib_streams,
                                     calib_horizon=calib_horizon,
                                     do_inference=do_inference,
                                     extrapolation_name=extrapolation_name,
                                     learn_kernel=learn_kernel, batch_size=batch_size,
                                     seeds_code=seed + rep)
                    if r["extrapolation"] is None:
                        do_inference = False        # gpflow missing: warn only once
                    rows.append({
                        "dataset": name, "objective": obj_name, "budget": budget,
                        "rep": rep, "detector": det_name, **_flat_row(r),
                    })
                print(f"  done {obj_name:8s} k={budget}  A_ids={rows[-1]['A_ids']}"
                      + (f"  rep={rep}" if n_rep > 1 else ""))

    _save(rows, dm, name, results_dir)
    _print_table(rows)
    return rows


def _flat_row(r):
    inf = r["extrapolation"] or {}
    return {
        "A": r["sensor"], "A_ids": r["sensor_ids"], "objective_value": r["val"],
        "threshold": r["cpd_param"].get("threshold"),
        "detection_delay": r["cpd"]["detection_delay"],
        "detection_delay_first": r["cpd"]["detection_delay_first"],
        "false_alarm_rate": r["cpd"]["false_alarm_rate"],
        "num_alarms": r["cpd"].get("num_alarms"),
        "no_alarm": r["cpd"].get("no_alarm"),
        "F1": r["cpd"]["F1"]["score"],
        "precision": r["cpd"]["F1"]["precision"],
        "recall": r["cpd"]["F1"]["recall"],
        "F1_sustained": (r["cpd"].get("F1_sustained") or {}).get("score"),
        "inference_MSE": inf.get("MSE"),
        "inference_coverage_95": inf.get("coverage_95"),
        "inference_n_eval": inf.get("n_eval"),
    }


def _save(rows, dm, name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    field = getattr(dm, "field", "na")
    for ch in " /()":
        field = field.replace(ch, "_" if ch in " /" else "")
    base = os.path.join(results_dir, f"benchmark_{name}_{field}")
    with open(base + ".json", "w") as f:
        json.dump({"describe": dm.describe(), "result": rows}, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    flat = []
    for r in rows:
        r = dict(r)
        r["A"] = " ".join(map(str, r["A"]))
        r["A_ids"] = " ".join(map(str, r["A_ids"]))
        flat.append(r)
    if flat:
        with open(base + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    print(f"\nResults saved to {base}.json / .csv")


def _print_table(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["objective"], r["budget"], r["detector"]), []).append(r)

    header = (f"{'objective':10s} {'k':>2s} {'detector':18s} {'thr':>8s} {'delay':>7s} "
              f"{'FAR':>7s} {'alarms':>7s} {'F1':>6s} {'MSE':>8s} {'cov95':>6s}")
    print("\n" + header)
    print("-" * len(header))
    for (obj, k, det), g in groups.items():
        print(f"{obj:10s} {k:2d} {det:18s} "
              f"{_fmt(_mean([r['threshold'] for r in g]), 8)} "
              f"{_fmt(_mean([r['detection_delay'] for r in g]), 7, 1)} "
              f"{_fmt(_mean([r['false_alarm_rate'] for r in g]), 7)} "
              f"{_fmt(_mean([r['num_alarms'] for r in g]), 7, 1)} "
              f"{_fmt(_mean([r['F1'] for r in g]), 6)} "
              f"{_fmt(_mean([r['inference_MSE'] for r in g]), 8, 4)} "
              f"{_fmt(_mean([r['inference_coverage_95'] for r in g]), 6, 2)}")
    print("\n(thresholds calibrated on the historical f0; 'random' rows are means over "
          "its repetitions; with no annotated change, delay is nan and a good detector "
          "raises 0 alarms; MSE/cov95 in the normalized scale at the observed evaluation "
          "entries only)")
