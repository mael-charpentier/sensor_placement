"""Declarative definition of the synthetic experiment grids.

Each GridSpec is one block of the sweep: a cross product of placement criteria,
detectors, kernels and change types, enumerated by
spgprcpd.experiments.synthetic.do_exp_general.
runs experiment id, counted across the enabled blocks in list order.

Two things follow from that and are worth stating plainly:

   Ids are positional.* They are assigned by walking grids in order and counting
    the cells of each enabled block.  Enabling, disabling or reordering a block renumbers
    every id after it, so results already in results/ no longer correspond to the same
    ids.  (The result files are named after the full configuration, not the id, so nothing is
    overwritten or mislabelled, but a job array resubmitted after an edit runs a different
    set of cells.)
   Disabled blocks are kept, not deleted.* They record which sweeps the study defines but
    did not run for the reported results.  enabled=False is the switch. Flipping it is how
    you reproduce or extend a part of the study.

This module deliberately imports nothing but the standard library, so the grid definitions
can be inspected and tested without gpflow/tensorflow installed.
"""
import itertools
from dataclasses import dataclass

# the experiment parameter vector
#: Names of the experiment parameters, in the order they appear throughout the synthetic
#: pipeline: the general_info axes, then the kernel_infos axes, then the four seeds.
#: This is the single source of truth, make_grid below, and pipeline /
#: verif_compute / get_name / the saved "param" entry in synthetic.py, all
#: follow it, and the figure scripts read it back through parse_params instead of
#: hardcoding positions.  Adding a parameter means adding it here and in those signatures.#: the assertion below catches a partial edit.
PARAM_FIELDS = (
    # general_info
    "N", "N_test", "time_sample", "num_to_choose", "change_time",
    "metric_class", "weighted", "extrapolation_class", "cpd_class", "batch_size",
    # kernel_infos
    "kernel_prior", "kernel_prior_type", "mu_prior", "change_format",
    "delta_kernel", "delta_mu", "kernel_learn", "kernel_learn_name",
    "use_time_serie_feature", "do_estimation", "extrapolation_class_est",
    "time_output", "calibrate_with_f1", "W",
    "do_extrapolation_result", "do_cpd_result",
    # seeds
    "seeds_point", "seeds_sensor", "seeds_init", "seeds_code",
)
assert len(PARAM_FIELDS) == 30, "PARAM_FIELDS is out of step with the pipeline signature"


def parse_params(param):
    """{name: value} for a result file's "param" entry.

    Accepts the current dict form and the older positional-list form, so result files
    written before PARAM_FIELDS existed still load.  Use this in every analysis or figure
    script: indexing param by hand is what silently mislabelled the earlier plots when
    parameters were appended to the pipeline.
    """
    if isinstance(param, dict):
        return dict(param)
    param = list(param)
    if len(param) != len(PARAM_FIELDS):
        raise ValueError(
            f"result file has {len(param)} parameters but this version of the pipeline "
            f"defines {len(PARAM_FIELDS)}; the file was written by a different version and "
            f"cannot be labelled reliably")
    return dict(zip(PARAM_FIELDS, param))


# building blocks
#: The four calibrated detectors compared head to head. All four get the same arl0 budget
#: of 500, so they differ only in the statistic.
DETECTORS_ALL = ["cusum_arl500", "cusum_no_post_arl500",
                 "ewma_arl500_a0.1", "ewma_arl500_a0.2"]
#: A single detector, for blocks whose question is about the placement or the gaussian process model, where
#: sweeping detectors would only multiply the run time.
DETECTORS_ONE = ["cusum_arl500"]

#: (time_sample, change_times, batch_size).  change_times is (tau1, tau2, tau3):
#: the historical record switches f0 to f1 at tau1 and back at tau2, and the held-out test
#: stream changes at tau3.  The long horizon pushes tau3 far out so the streaming models have
#: many batches of in-control data before the change.
SHORT_HORIZON = (250, (70, 120, 170), 50)
LONG_HORIZON = (1000, (70, 120, 640), 40)

#: The three gaussian process models compared in the study: the streaming-sparse approximation, and the
#: two exact variants that differ in how often they refit (see inference/exact.py).
#:
#: extrapolation_class is embedded in the result filename by get_name, so these
#: strings are what a completed sweep is keyed on.  exact_no_retrain was renamed once,
#: so results produced before that rename are keyed on the previous string and will not be
#: recognised as already done.
MODELS = ["streaming_sparse", "exact", "exact_no_retrain"]


@dataclass(frozen=True)
class GridSpec:
    """One block of the sweep.

    Attributes:
    name: str
        Short identifier used in logs and when talking about the sweep.
    enabled: bool
        Whether do_exp walks this block.  See the module docstring on renumbering.
    general_info, kernel_infos: list
        The two grid halves consumed by do_exp_general, kept in exactly the layout it
        expects (see synthetic.PARAM_FIELDS for what each position means).
    note: str
        Why the block exists, or why it is switched off.
    """
    name: str
    enabled: bool
    general_info: list
    kernel_infos: list
    note: str = ""

    @property
    def n_cells(self):
        """Number of (general x kernel) combinations, before verif_compute filtering."""
        n_general = 1
        for axis in self.general_info:
            n_general *= len(axis)
        return n_general * len(self.kernel_infos)


def make_grid(*, horizon, criteria, weights, model, detectors,
              learn_kernel=("init_learning",), estimate_f0=True,
              calibrate_with_f1=(False,), score_inference=True, score_detection=False,
              time_feature=0):
    """Build the (general_info, kernel_infos) pair for one block.

    Only the axes that actually vary between blocks are parameters. Everything else is fixed
    across the whole study and written literally below.

    Parameters:
    horizon: tuple
        SHORT_HORIZON or LONG_HORIZON.
    criteria, weights: sequence
        Placement criteria and the weight alpha.  weights is only meaningful for the
        "weighted" criterion (alpha = 0 is pure kl, 1 is pure mutual information). The other criteria
        ignore it, and 1.1 is the conventional "not applicable" filler.
    model: str
        The gaussian process model, used both for the online inference and for the f0 estimation stage
        (the two are the same in every block of this study).
    detectors: sequence
        DETECTORS_ALL or DETECTORS_ONE.
    learn_kernel: sequence
        Kernel the inference model starts from.  "init_learning" is the deliberately
        wrong starting point that has to be fitted. The "diff_*" names hand the model a
        specific true kernel, which is what exact_no_train needs.
    estimate_f0: bool
        Whether f0 is estimated from the historical record (the realistic setting) or taken
        as known (the reference setting).
    calibrate_with_f1: sequence of bool
        False calibrates the detector threshold on f0 alone. True also uses f1 to pick the
        operating point.
    score_inference, score_detection: bool
        Which halves of the pipeline are evaluated.  Blocks usually score one or the other:
        computing both doubles the run time for a question that only concerns one.
    time_feature: int
        0 = time as an input feature, 1 = random walk, 2 = learn without the time dimension.
    """
    time_sample, change_times, batch_size = horizon
    general_info = [
        [10],                       # N: candidate sensor locations
        [5],                        # N_test: held-out evaluation locations
        [time_sample],
        [5],                        # num_to_choose: the placement budget b
        [change_times],
        list(criteria),
        list(weights),
        [model],
        list(detectors),
        [batch_size],
    ]
    kernel_infos = list(itertools.product(
        ["Matern_Anisotropic"],                         # kernel_prior
        ["diff_3", "diff_4", "diff_5"],                 # kernel_prior_type: 3 true kernels
        ["zero"],                                       # mu_prior
        [True],                                         # change_format: delta given as a kernel
        [None, ["Matern_Anisotropic", "random"]],       # delta_kernel: no covariance change / random
        ["propagation_true_kernel", "no_change"],       # delta_mu: spatial mean shift / none
        [None],                                         # kernel_learn: same family as the prior
        list(learn_kernel),                             # kernel_learn_name
        [time_feature],                                 # use_time_serie_feature
        [estimate_f0],                                  # do_estimation
        [model],                                        # extrapolation_class_est
        [time_feature == 2],                            # time_output
        list(calibrate_with_f1),
        [20],                                           # W: detection window for the score
        [score_inference],                              # do_extrapolation_result
        [score_detection],                              # do_cpd_result
    ))
    return general_info, kernel_infos


def _spec(name, enabled, note, **kwargs):
    general_info, kernel_infos = make_grid(**kwargs)
    return GridSpec(name=name, enabled=enabled, general_info=general_info,
                    kernel_infos=kernel_infos, note=note)


# the sweep
# order is significant: experiment ids are assigned by walking this list. See the module
# docstring.
GRIDS: list[GridSpec] = [
    # detection-scored blocks on the exact gaussian process, all five detectors
    _spec("detection/weighted", False,
          "Detection quality along the weight between mutual information and kl, with "
          "every detector. Superseded by the inference-scored model comparison below, "
          "which is what the reported results use.",
          horizon=SHORT_HORIZON, criteria=["weighted"], weights=[0.0, 0.5, 1.0],
          model="exact", detectors=DETECTORS_ALL,
          calibrate_with_f1=(False, True), score_inference=False, score_detection=True),
    _spec("detection/min_MI+max_H", False,
          "Same, for the two min and max criteria. The weight does not apply to them, so "
          "1.1 is the filler.",
          horizon=SHORT_HORIZON, criteria=["min_MI", "max_H"], weights=[1.0, 1.1],
          model="exact", detectors=DETECTORS_ALL,
          calibrate_with_f1=(False, True), score_inference=False, score_detection=True),
    _spec("detection/kl+entropy", False,
          "Same, for the two single-objective criteria.",
          horizon=SHORT_HORIZON, criteria=["kl_quadratic", "entropy"], weights=[1.1],
          model="exact", detectors=DETECTORS_ALL,
          calibrate_with_f1=(False, True), score_inference=False, score_detection=True),
]

# inference-scored blocks, once per gaussian process model. This is the part of the sweep the
# reported results come from, and the question it answers is how the reconstruction error
# varies along the weight between mutual information and kl, for each model in turn. The two
# companion blocks per model, the ones holding the other criteria, are defined but switched
# off.
for _model in MODELS:
    GRIDS += [
        _spec(f"inference/{_model}/weighted", True,
              "reported. Reconstruction error along the weight between mutual information "
              "and kl, on the long horizon so that the streaming model sees many "
              "in-control batches before the change.",
              horizon=LONG_HORIZON, criteria=["weighted"], weights=[0.0, 0.5, 1.0],
              model=_model, detectors=DETECTORS_ONE),
        _spec(f"inference/{_model}/min_MI+max_H", False,
              "The min and max criteria on the short horizon. Enable it to extend the "
              "comparison.",
              horizon=SHORT_HORIZON, criteria=["min_MI", "max_H"], weights=[1.0, 1.1],
              model=_model, detectors=DETECTORS_ONE),
        _spec(f"inference/{_model}/kl+entropy", False,
              "The single-objective criteria. Enable it to extend the comparison.",
              horizon=LONG_HORIZON, criteria=["kl_quadratic", "entropy"], weights=[1.1],
              model=_model, detectors=DETECTORS_ONE),
    ]

del _model


def enabled_grids():
    """The blocks do_exp walks, in id order."""
    return [g for g in GRIDS if g.enabled]


def describe():
    """One line per block: whether it runs, and how many cells it contributes."""
    lines = [f"{'block':38s} {'on':>3s} {'cells':>7s}  note"]
    lines.append("-" * 100)
    for g in GRIDS:
        lines.append(f"{g.name:38s} {'yes' if g.enabled else ' no':>3s} "
                     f"{g.n_cells:7d}  {g.note.splitlines()[0] if g.note else ''}")
    total = sum(g.n_cells for g in enabled_grids())
    lines.append("-" * 100)
    lines.append(f"{'enabled total, upper bound on the ids':38s} {'':>3s} {total:7d}")
    lines.append("upper bound because verif_compute drops inconsistent combinations at run time")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
