"""Synthetic benchmark: the full placement -> calibration -> detection -> inference pipeline
on data sampled from a known f0 / f1.

Knowing both regimes is what the synthetic study buys: the f1-dependent placement criteria
(kl, the weighted mutual information+kl family) are available here, and every estimate can
be scored against the true generating parameters.  The real-data engine
(spgprcpd.experiments.real) is the f0-only counterpart.

One grid cell per call, so a job array can fan the ids out over machines:

    from spgprcpd.experiments import synthetic
    synthetic.do_exp(0)
"""
import copy
import gc
import hashlib
import itertools
import json
import os
import time

import numpy as np
import tensorflow as tf

from spgprcpd import paths
from spgprcpd.data import gaussian, kernels
from spgprcpd.data import synthetic as synthetic_data
from spgprcpd.detection import bocpd, cusum, ewma
from spgprcpd.experiments import grids
from spgprcpd.inference import exact, sparse
from spgprcpd.placement import entropy, kl_divergence, mutual_information

np.random.seed(0)

do_verif_func = True

# The experiment parameter vector and its parser live in `grids` (stdlib-only, so they
# are importable and testable without the gaussian process stack). Re-exported here because this is
# where the pipeline that follows the schema lives.
PARAM_FIELDS = grids.PARAM_FIELDS
parse_params = grids.parse_params


def pipeline(N, N_test, time_sample, num_to_choose, change_times, metric_class, weighted,
             extrapolation_class, cpd_class, batch_size, kernel_prior,
             kernel_prior_type, mu_prior, change_format, delta_kernel, delta_mu,
             kernel_learn, kernel_learn_name, use_time_serie_feature, do_estimation, extrapolation_class_est,
             time_output, calibrate_with_f1, W,
             do_extrapolation_result=True, do_cpd_result=True,
             seeds_point=None, seeds_sensor=0, seeds_init=0, seeds_code=0,):

    np.random.seed(seeds_init)
    time_begin = time.time_ns()

    if change_times is None:
        change_time, hist_change_times = None, None
    elif len(change_times) == 3:
        change_time = change_times[2]
        hist_change_times = (change_times[0], change_times[1])
    else:
        change_time = change_times[0]
        hist_change_times = None

    if delta_kernel is None and delta_mu is None:
        change_time = None


    time_serie_feature = "time" if use_time_serie_feature == 0 else "no_time"
    time_serie_feature_learn = "learn_not_time" if use_time_serie_feature == 2 else ("time" if use_time_serie_feature == 0 else "no_time")

    kernel = kernels.kernels[kernel_prior][time_serie_feature][kernel_prior_type]()

    size_param = 1 if use_time_serie_feature == 2 else (time_sample if use_time_serie_feature == 0 else 1)
    mu = mus[mu_prior]((N + N_test) * size_param)
    dist_prior = gaussian.Multivariate_Gaussian(mu, kernel)

    if change_time is not None:
        mu = mus[mu_prior]((N + N_test) * size_param)
        dist_prior_complet = gaussian.Multivariate_Gaussian(mu, kernel)

        # Built for its side effect, not its value, do not "clean up" this call.
        # Synthetic_Data_Manager.init calls update_points(V) on the distribution it is
        # handed, and that is what populates dist_prior_complet.points, read below by
        # deltas_sigma / deltas_mu. Without it those receive None.
        # do_sample=False: only the sensor locations are wanted here, not a realisation.
        synthetic_data.Synthetic_Data_Manager(
            time_sample, N, dist_prior_complet, None, None,
            hist_change_times=hist_change_times, N_test=N_test, num_features=2,
            seed=seeds_sensor, seeds_point=seeds_point,
            random_walk=use_time_serie_feature, do_sample=False)

        if delta_kernel is None:
            kernel_post = copy.deepcopy(kernel)
        else:
            if change_format:
                kernel_post_name = delta_kernel[0]
                if kernel_post_name is None:
                    kernel_post_name = kernel_prior
                kernel_post = kernels.kernels[kernel_post_name][time_serie_feature][delta_kernel[1]]()
            else:
                sigma = dist_prior_complet.Sigma()
                d_s = deltas_sigma[delta_kernel](dist_prior_complet.points, size_param)
                d_s = (d_s + d_s.T)/2
                kernel_post = sigma + d_s

        mu = mus[mu_prior]((N + N_test) * size_param)
        if delta_mu is None:
            d_mu = 0
        else:
            d_mu = deltas_mu[delta_mu](dist_prior_complet.points, size_param, kernel)
        mu_post = mu + d_mu

        dist_post = gaussian.Multivariate_Gaussian(mu_post, kernel_post)
    else:
        dist_post = None

    if kernel_learn is None:
        kernel_learn = kernel_prior
    kernel_init = kernels.kernels[kernel_learn][time_serie_feature_learn][kernel_learn_name]()
    kernel_init_cpd = kernels.kernels[kernel_learn][time_serie_feature][kernel_learn_name]()

    cpd_cls = register_cpd[cpd_class]
    cpd_method = lambda dist_prior=dist_prior, dist_post=dist_post, kernel_init_cpd=kernel_init_cpd: cpd_cls(dist_prior, dist_post, kernel_init_cpd)

    extrapolation_cls = register_extra[extrapolation_class]
    extrapolation_cls_est = None
    if do_estimation:
        if extrapolation_class_est is None:
            extrapolation_cls_est = extrapolation_cls
        else:
            extrapolation_cls_est = register_extra[extrapolation_class_est]

    data_manager = synthetic_data.Synthetic_Data_Manager(time_sample, N, dist_prior,
                                                         dist_post, change_time, hist_change_times=hist_change_times, N_test=N_test, num_features=2, seed=seeds_sensor,
                                                         seeds_point=seeds_point, random_walk=use_time_serie_feature,
                                                         do_estimation=do_estimation,
                                                         extrapolation_cls=extrapolation_cls_est,
                                                         kernel_prior_init=kernels.kernels[kernel_learn][time_serie_feature_learn][kernel_learn_name](),
                                                         kernel_post_init=kernels.kernels[kernel_learn][time_serie_feature_learn][kernel_learn_name](),
                                                         est_time_output=time_output
                                                        )

    # every factory in register_metric takes the same four arguments so the registry stays
    # uniform. The criteria shipped here only read the first one, the weight
    metric = register_metric[metric_class](weighted, cpd_method, data_manager, W)

    time_setup = time.time_ns() - time_begin
    time_begin = time.time_ns()

    np.random.seed(seeds_code)
    A, val_A = metric(num_to_choose, list(range(N)), list(range(N, N+N_test)), dist_prior, dist_post)
    #print(A)
    B = [i for i in range(N) if i not in A]

    time_metric = time.time_ns() - time_begin
    time_begin = time.time_ns()

    # The oracle eval model (exact_no_train) uses the true kernel, whose hyperparameters live
    # in the raw data scale, so it must be evaluated on raw (un-normalized) data.  The other
    # models retrain and adapt to the normalized scale, so they keep normalize=True.
    normalize_extra = extrapolation_class != "exact_no_train"

    np.random.seed(seeds_code)
    resultats = data_manager.get_dataset(A, B, time_dimension=False, normalize=normalize_extra, batch_size=batch_size, time_output=time_output, save_norm=True)
    np.random.seed(seeds_code)
    resultats_iid = data_manager.get_dataset(A, B, time_dimension=True, normalize=False, batch_size=batch_size)

    time_get_data = time.time_ns() - time_begin

    time_extra = 0
    time_cpd = 0


    np.random.seed(seeds_code)
    # The eval model normally learns its hyperparameters from kernel_init (Exact_GP retrains
    # it).  A no-training model (exact_no_train, used for the oracle / true-model reference)
    # cannot, so it must be given a fresh copy of the true data-generating kernel, otherwise
    # it predicts with the untrained init kernel and its mean squared error/coverage reflect that kernel, not
    # the placement (this is the report' "benchmark = true model").  A fresh instance avoids
    # mutating dist_prior's kernel.
    if extrapolation_class == "exact_no_train":
        eval_kernel = kernels.kernels[kernel_prior][time_serie_feature][kernel_prior_type]()
        # a retraining model learns its observation noise. The no-training model can't, so
        # give it the true generator noise too (sigma_eps^2), otherwise its 1e-3 default is
        # too small and it stays overconfident (undercoverage) even with the true kernel.
        true_noise = data_manager.noise_std ** 2 if getattr(data_manager, "noise_std", None) else 1e-3
        extrapolation = extrapolation_cls(eval_kernel, noise_variance=true_noise)
    else:
        extrapolation = extrapolation_cls(kernel_init)
    time_begin = time.time_ns()
    cpd_result = None
    cpd_param = None
    if change_time is not None and do_cpd_result:
        np.random.seed(seeds_code)
        cpd = cpd_method()
        # A criterion that had to score detection performance while searching has already
        # calibrated a threshold for A, so it can hand it back through get_threshold and
        # save the work. None of the criteria shipped here do, so in practice this falls
        # through to the calibration below.
        reused = getattr(metric, "get_threshold", lambda *a: None)(A)
        if reused is not None:
            cpd.threshold = reused
            cpd._deferred_calibration = False
        else:
            # calibrate on the actual stream the detector will run on: its f0 portion (and,
            # when calibrate_with_f1, the full f0 to f1 stream for the expected detection delay).
            tau = hist_change_times[0]
            stream = data_manager.get_dataset(A, B, time_dimension=True, normalize=False, batch_size=None, phase='historical')[0][2]                 # y_iid_train: the CPD stream at A
            f0_data = stream[:tau] if (tau is not None and 1 < tau <= len(stream)) else stream
            if calibrate_with_f1:
                cpd.calibrate_on_f0_f1(B, A, f0_data=f0_data, chg_data=stream, tau=tau,
                                       W=W, n_streams=1, horizon=300)
            else:
                cpd.calibrate_on_f0(B, A, f0_data=f0_data, n_streams=1, horizon=300)

            cpd_param = cpd.get_all_params()
    time_calibrate_cpd = time.time_ns() - time_begin

    np.random.seed(seeds_code)
    for i in range(len(resultats)):
        X_train, y_train, _, X_test, _, _ = resultats[i]
        _, _, y_iid_train, _, _, _ = resultats_iid[i]

        if do_extrapolation_result:
            time_begin = time.time_ns()

            extrapolation.update(X_train, y_train)
            extrapolation.predict(X_test)

            time_extra += time.time_ns() - time_begin

        if change_time is not None and do_cpd_result:
            time_begin = time.time_ns()
            cpd.alarms(B, A, y_iid_train)
            time_cpd += time.time_ns() - time_begin

    extrapolation_result = None
    if do_extrapolation_result:
        time_begin = time.time_ns()

        np.random.seed(seeds_code)
        _, _, _, _, y_test, _ = data_manager.get_dataset(A, B, time_dimension=False, normalize=normalize_extra, batch_size=None, save_norm=True)[0]
        extrapolation_result = extrapolation.eval(y_test, dist_prior, dist_post)

        time_extra += time.time_ns() - time_begin

    if change_time is not None and do_cpd_result:
        time_begin = time.time_ns()

        np.random.seed(seeds_code)
        cpd_result = cpd.eval(data_manager.test_change_time)

        time_cpd += time.time_ns() - time_begin

    return {
            "extrapolation": extrapolation_result,
            "cpd": cpd_result,
            "cpd_param": cpd_param,
            "sensor": A,
            "val": val_A,
#            "history": {
#                "extrapolation": extrapolation._history,
#                "cpd": (cpd._history if change_time is not None else None)
#            },
            "time": {
                "setup": time_setup,
                "metric": time_metric,
                "data": time_get_data,
                "calibration_cpd": time_calibrate_cpd,
                "extrapolation": time_extra,
                "cpd": time_cpd
            }
        }

def propagation_delta_mu(V, strength=2.0, decay=1.0, kernel=None):
    V = np.array(V)  # shape (n, 2)

    # Step 1: pick a random index as the source
    idx = np.random.randint(len(V))
    source = V[idx:idx+1]  # keep 2D shape for kernel computations

    # Step 2 & 3: compute deltas
    if kernel is not None:
        kernel_dim = kernel.__class__(
            lengthscales=kernel.lengthscales.numpy()[:2], #:3
            variance=kernel.variance,
        )
        # Compute kernel between all points and the source
        # GPflow expects float64
        V_tf = tf.convert_to_tensor(V, dtype=tf.float64)
        source_tf = tf.convert_to_tensor(source, dtype=tf.float64)
        deltas_tf = kernel_dim.K(V_tf, source_tf)
        deltas = strength * deltas_tf.numpy().flatten()
    else:
        # Default: exponential decay
        distances = np.linalg.norm(V - source, axis=1)
        deltas = strength * np.exp(-decay * distances)

    return deltas


def verif_compute(N, N_test, time_sample, num_to_choose, change_time, metric_class,
                  weighted, extrapolation_class, cpd_class, batch_size,
                  kernel_prior, kernel_prior_type, mu_prior, change_format,
                  delta_kernel, delta_mu, kernel_learn, kernel_learn_name, use_time_serie_feature, do_estimation, extrapolation_class_est, time_output, calibrate_with_f1, W, do_extrapolation_result=True, do_cpd_result=True):
    if (delta_kernel is None and delta_mu == "no_change"):
        return False
    if (metric_class == "kl_quadratic" and delta_mu == "no_change"):
        return False
    if (kernel_learn_name != kernel_prior_type and kernel_learn_name != "init_learning"):
        return False
    return True

def do_all_seed(name_seed, seed, N, N_test, time_sample, num_to_choose, change_time, metric_class,
                  weighted, extrapolation_class, cpd_class, batch_size,
                  kernel_prior, kernel_prior_type, mu_prior, change_format,
                  delta_kernel, delta_mu, kernel_learn, kernel_learn_name, use_time_serie_feature, do_estimation, extrapolation_class_est, time_output, calibrate_with_f1, W, do_extrapolation_result=True, do_cpd_result=True, **kwargs):
    return True

def get_name(N, N_test, time_sample, num_to_choose, change_time, metric_class, weighted,
             extrapolation_class, cpd_class, batch_size, kernel_prior,
             kernel_prior_type, mu_prior, change_format, delta_kernel, delta_mu,
             kernel_learn, kernel_learn_name, use_time_serie_feature, do_estimation, extrapolation_class_est, time_output, calibrate_with_f1, W,
             do_extrapolation_result, do_cpd_result, seeds_point, seeds_sensor,
             seeds_init, seeds_code):
    result_filename = f"{seeds_point}_{seeds_sensor}_{seeds_init}_{seeds_code}"
    result_filename += f"_{batch_size}"
    result_filename += f"_{N}_{N_test}"
    result_filename += f"_{time_sample}"
    result_filename += f"_{num_to_choose}"
    result_filename += f"_{change_time}"
    result_filename += f"_{use_time_serie_feature}_{kernel_prior}_{kernel_prior_type}"
    result_filename += f"_{do_estimation}_{extrapolation_class_est}_{time_output}"
    result_filename += f"_{mu_prior}"
    result_filename += f"_{change_format}_{delta_kernel}_{delta_mu}"
    result_filename += f"_{kernel_learn}_{kernel_learn_name}_{calibrate_with_f1}_{W}"
    result_filename += f"_{do_extrapolation_result}_{do_cpd_result}"
    result_filename += f"_{metric_class}_{weighted}_{extrapolation_class}_{cpd_class}"
    result_filename += ".json"

    # ext4 caps filenames at 255 bytes.  Names that fit are kept unchanged (so existing
    # result files still match). Longer ones keep a readable prefix and are disambiguated
    # by a hash of the full name, the exact config remains in the json's "param" field.
    if len(result_filename.encode()) > 255:
        h = hashlib.md5(result_filename.encode()).hexdigest()[:12]
        result_filename = result_filename[:190] + "___" + h + ".json"

    return result_filename

def run_seed(test_seeds, id, *args, **kwargs):
    if len(test_seeds) == id:
        try:
            result_pipeline = pipeline(*args, **kwargs)
        except Exception as ee:
            print(ee)
            result_pipeline = {
                "extrapolation": None,
                "cpd": None,
                "cpd_param": None,
                "sensor": None,
                "val": None,
#                "history": {
#                    "extrapolation": extrapolation._history,
#                    "cpd": (cpd._history if change_time is not None else None)
#                },
                "time": {
                    "setup": None,
                    "metric": None,
                    "data": None,
                    "calibration_cpd": None,
                    "extrapolation": None,
                    "cpd": None
                }
            }

        # release memory
        tf.keras.backend.clear_session()  # releases tensorflow graphs
        gc.collect()  # force Python garbage collection

        return result_pipeline
    name_seed, seeds = test_seeds[id]

    all_results = []
    if seeds < 0:
        for seed in range(-seeds):
            if do_verif_func and not do_all_seed(name_seed, seed, *args, **kwargs):
                continue
            kwargs[name_seed] = seed
            r = run_seed(test_seeds, id+1, *args, **kwargs)
            if r is not None:
                all_results.append(r)
    else:
        kwargs[name_seed] = seeds
        r = run_seed(test_seeds, id+1, *args, **kwargs)
        if r is not None:
            all_results.append(r)

    return all_results


mus = {
    "zero": lambda x: np.zeros(x)
}

register_metric = {
    "entropy": lambda x, cpd, data, W: entropy.Entropy(),
    "min_MI": lambda x, cpd, data, W: mutual_information.Min_MI(percent_substitution = x),
    "max_H": lambda x, cpd, data, W: entropy.Max_Entropy(percent_substitution = x),
    "weighted": lambda x, cpd, data, W: x * mutual_information.Mutual_Information() + (1-x) * kl_divergence.KL_Divergence_Exhaustive(), # (kl_divergence.KL_Divergence if x == 0 else kl_divergence.KL_Divergence_Exhaustive)
    "kl_quadratic": lambda x, cpd, data, W: kl_divergence.KL_Divergence(),
}

hazards = [1/2, 1/5, 1/10]          # for the bocpd hazard

register_cpd = {}
for arl in [200, 500, 1000]:
    # likelihood ratio cusum. Needs f1, so it only exists on synthetic data
    register_cpd[f"cusum_arl{arl}"] = (
        lambda dist_prior, dist_post, kernel_init, arl=arl:
        cusum.CuSum(dist_prior, dist_post, alpha_arl=arl)
    )
    # normalised cusum, the f0-only counterpart
    register_cpd[f"cusum_no_post_arl{arl}"] = (
        lambda dist_prior, dist_post, kernel_init, arl=arl:
        cusum.CuSum_Without_Post(dist_prior, dist_post, alpha_arl=arl)
    )
    for alpha in [0.1, 0.2]:
        register_cpd[f"ewma_arl{arl}_a{alpha}"] = (
            lambda dist_prior, dist_post, kernel_init, arl=arl, alpha=alpha:
            ewma.EWMA(alpha, dist_prior, dist_post, alpha_arl=arl)
        )
    for h in hazards:
        # k=5 and not the default 0, because at k=0 the bocpd alarm statistic is
        # P(r_t = 0), which a constant hazard makes equal to the hazard whatever the data
        # does. See the note in detection/bocpd.py
        register_cpd[f"bocpd_arl{arl}_h{h}"] = (
            lambda dist_prior, dist_post, kernel_init, arl=arl, h=h:
            bocpd.BOCPD(dist_prior, lambda r, h=h: h, alpha_arl=arl, k=5)
        )

# The three exact variants form a ladder differing only in how often the kernel is
# fitted: every batch / first batch only / never (see inference/exact.py).
register_extra = {
    "streaming_sparse":       sparse.Streaming_Sparse_GP,
    "exact":                  exact.Exact_GP,                  # fit every batch
    "exact_no_retrain":       exact.Exact_GP_no_retraining,    # fit the first batch only
    "exact_no_train":         exact.Exact_GP_no_training,      # never fit (oracle)
}

deltas_mu = {
    "no_change": lambda V, size_param, K: np.zeros(len(V) * size_param),
    "add 1": lambda V, size_param, K: np.ones(len(V) * size_param),
    "N(0,0.1)": lambda V, size_param, K: np.random.normal(0, 0.1, len(V) * size_param),
    "N(0,1)": lambda V, size_param, K: np.random.normal(0, 1, len(V) * size_param),
    "N(5,5)": lambda V, size_param, K: np.random.normal(5, 5, len(V) * size_param),
    "one_change": lambda V, size_param, K: np.tile(np.random.permutation([0]*(len(V)-1) + [2]), size_param),
    "propagation": lambda V, size_param, K: np.tile(propagation_delta_mu(V, strength=np.random.normal(3,1), kernel=None), size_param),
    "propagation_true_kernel": lambda V, size_param, K: np.tile(propagation_delta_mu(V, strength=np.random.normal(3,1), kernel=K), size_param)
}
deltas_sigma = {
    "no_change": lambda V, size_param: np.zeros((len(V) * size_param, len(V) * size_param)),
    "N(0,1e-5)": lambda V, size_param: np.random.normal(0, 1e-5, (len(V) * size_param, len(V) * size_param)),
    "N(0,0.01)": lambda V, size_param: np.random.normal(0, 0.01, (len(V) * size_param, len(V) * size_param)),
    "N(0.2,0.2)": lambda V, size_param: np.random.normal(0.2, 0.2, (len(V) * size_param, len(V) * size_param)),
}


result_folder = str(paths.RESULTS_DIR) + os.sep


def do_exp_general(do_id, seeds_point, seeds_sensor, seeds_init, seeds_code, general_info, kernel_infos):
    exp_id = -1
    print("LOOK FOR:",do_id)

    for args in itertools.product(*general_info):
        for kernel_info in kernel_infos:
            if do_verif_func and not verif_compute(*args, *kernel_info):
                continue

            exp_id += 1
            if do_id != exp_id:
                continue

            result_filename = get_name(*args, *kernel_info, seeds_point, seeds_sensor, seeds_init, seeds_code)

            os.makedirs(result_folder, exist_ok=True)
            result_filename = result_folder + result_filename
            if not os.path.exists(result_filename):
                print(f"Do this file {result_filename}")
                time_begin = time.time_ns()
                all_results = run_seed(
                    [
                        ("seeds_point", seeds_point),
                        ("seeds_sensor", seeds_sensor),
                        ("seeds_init", seeds_init),
                        ("seeds_code", seeds_code)
                    ], 0,
                    *args,
                    *kernel_info
                )
                time_end = time.time_ns() - time_begin
                if len(all_results) > 0:
                    result = {
                        "result": all_results,
                        "param": parse_params([*args, *kernel_info, seeds_point,
                                               seeds_sensor, seeds_init, seeds_code]),
                    }
                    with open(result_filename, "w") as _f:
                        json.dump(result, _f, indent=4, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
                    print(f"Result saved to {result_filename}, in {time_end * 1e-9:.3f}s")
                else:
                    print(f"No result in {result_filename}, in {time_end * 1e-9:.3f}s")

                # release memory
                tf.keras.backend.clear_session()  # releases tensorflow graphs
                gc.collect()  # force Python garbage collection
            return True, None

    print(exp_id)
    return False, exp_id

def do_exp(exp_id):
    """Run experiment number exp_id of the sweep, then return.

    Ids are counted across the enabled blocks of spgprcpd.experiments.grids.GRIDS, in
    list order: block 0 owns ids 0.n0-1, the next enabled block owns n0.n0+n1-1, and so on.
    Each block is offered the remaining id and reports back how many it consumed, so the
    offset accumulates, which is why enabling or reordering a block renumbers everything
    after it (see the grids module docstring).

    Returns True if some block owned exp_id (it ran, or was skipped because its result
    file already exists), False if exp_id is past the end of the sweep.
    """
    seeds_point = 0     # sensor locations
    seeds_sensor = -5   # the sampled data (negative -N -> the seed set 0.N-1)
    seeds_init = -3     # setup randomness, including the change delta
    seeds_code = 0      # everything else, for reproducibility

    remaining = exp_id
    consumed = -1
    for spec in grids.enabled_grids():
        remaining -= consumed + 1
        finish, consumed = do_exp_general(remaining, seeds_point, seeds_sensor, seeds_init,
                                          seeds_code, spec.general_info, spec.kernel_infos)
        if finish:
            return True

    print(f"id {exp_id} is past the end of the sweep")
    return False
