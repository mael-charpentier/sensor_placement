"""Data manager for the real Intel Berkeley Lab record, dataset/labdata, from Krause et al.

Self-contained loader, no external data.load_data module needed. It reads two files:

  * data.txt.gz, one reading per row, as date, time, epoch, moteid, temperature, humidity,
    light and voltage. Around 2.3 million rows at roughly 31 second epochs, with many gaps
    and bad readings.
  * mote_locs.txt, as moteid, x, y, the fixed mote coordinates in metres in the lab frame.

The loader drops physically impossible readings, see VALID_RANGE, resamples each mote onto a
common regular grid taking the mean per bin with a default of 15 minutes, keeps the requested
motes or else the best-covered ones, interpolates the internal gaps, and hands the dense
(T, n) block to spgprcpd.data.real_base.Real_Data_Manager_Base, which exposes the exact
interface of the synthetic manager, meaning V, X and y and y_iid, the empirical dist_prior
and dist_post, and the historical and test phases.

Example:
    from spgprcpd.data.labdata import Labdata_Data_Manager
    dm = Labdata_Data_Manager(field="temperature", num_points=10, N_test=4,
                              time_range=("2004-03-01", "2004-03-10"))
    print(dm.describe())
"""
import os

import numpy as np
import pandas as pd

from spgprcpd import paths
from spgprcpd.data.real_base import Real_Data_Manager_Base, densify

#: Resolved centrally (see spgprcpd.paths). Override with the SPGPRCPD_DATA env var.
DATA_DIR = str(paths.LABDATA_DIR)

FIELDS = ["temperature", "humidity", "light", "voltage"]

# physically plausible ranges. Readings outside (battery deaths, glitches) become nan
VALID_RANGE = {
    "temperature": (0.0, 50.0),
    "humidity": (0.0, 100.0),
    "light": (0.0, 5000.0),
    "voltage": (1.5, 3.5),
}


def load_labdata_long(data_dir=None, max_rows=None, clip=True):
    """Long-format Intel-Lab record: columns time (datetime64), sensor (mote id) and
    the four channels of FIELDS (nan when missing or, with clip, implausible)."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    df = pd.read_csv(os.path.join(data_dir, "data.txt.gz"), sep=r"\s+", header=None,
                     names=["date", "clock", "epoch", "sensor"] + FIELDS,
                     nrows=max_rows, engine="c")
    df["time"] = pd.to_datetime(df["date"] + " " + df["clock"], errors="coerce")
    df["sensor"] = pd.to_numeric(df["sensor"], errors="coerce")
    df = df.dropna(subset=["time", "sensor"])
    df["sensor"] = df["sensor"].astype(int)
    for f in FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        if clip:
            lo, hi = VALID_RANGE[f]
            df.loc[(df[f] < lo) | (df[f] > hi), f] = np.nan
    return df[["time", "sensor"] + FIELDS].sort_values(["sensor", "time"]).reset_index(drop=True)


def load_mote_locations(data_dir=None):
    """{mote_id: (x, y)} in metres (lab frame) from mote_locs.txt."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    locs = np.loadtxt(os.path.join(data_dir, "mote_locs.txt"))
    return {int(m): (float(x), float(y)) for m, x, y in locs}


def pivot_field(df, field, resample="15min", sensors=None, time_range=None):
    """Wide (T_bins, n_motes) DataFrame of one channel: mean per resample bin per
    mote, on the common regular grid spanning the (cropped) record."""
    if field not in FIELDS:
        raise KeyError(f"unknown field {field!r}; available: {FIELDS}")
    sub = df.dropna(subset=[field])
    if sensors is not None:
        sub = sub[sub["sensor"].isin(list(sensors))]
    if time_range is not None:
        start, end = time_range
        if start is not None:
            sub = sub[sub["time"] >= pd.Timestamp(start)]
        if end is not None:
            sub = sub[sub["time"] <= pd.Timestamp(end)]
    if sub.empty:
        raise ValueError(f"no data left for field {field!r} with the requested "
                         f"sensors/time_range")
    wide = (sub.set_index("time")
               .groupby("sensor")[field]
               .resample(resample).mean()
               .unstack(level="sensor"))
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=resample)
    return wide.reindex(full)


class Labdata_Data_Manager(Real_Data_Manager_Base):
    """Real-data manager for the Intel Berkeley Lab network, see the module docstring.

    Parameters:
    field: str
        One of "temperature", "humidity", "light", "voltage".
    sensors: list of int, or None
        Mote ids to use, order kept. None picks the best-covered ones.
    exclude_sensors: iterable of int
        Mote ids to always drop, applied before any selection.
    random_selection: bool
        With sensors=None, pick the num_points + N_test motes uniformly at random among
        those passing min_coverage, including which of them become the test motes. The
        draw is reproducible through seed, so several random settings can be benchmarked.
    num_points: int or None
        With sensors=None, keep that many best-covered motes on top of N_test. None keeps
        every mote passing min_coverage.
    N_test: int
        Held-out test motes, taken as the last columns.
    time_range: (start, end) or None
        Crop, as timestamps or strings. An open side is written as None.
    resample: str
        Bin width of the common grid, a pandas offset such as "15min".
    min_coverage: float
        Drop a mote observed in fewer than this fraction of the bins, counted before the
        interpolation.
    split_time: int or timestamp, or None
        Boundary between the historical and the test phase, half of T by default. The whole
        historical phase is f0, since real data never exposes f1, so dist_post is None.
    change_time: int or timestamp, or None
        A real change annotated in the test stream. None, the default, assumes no change,
        and the detectors are then scored on staying quiet.
    max_rows: int or None
        Cap on the rows read out of the log, which is large.
    max_samples: int or None
        Cap on T, by subsampling evenly.
    """

    def __init__(self, field="temperature", *, sensors=None, num_points=12, N_test=0,
                 exclude_sensors=(), random_selection=False, time_range=None,
                 resample="15min", min_coverage=0.5,
                 split_time=None, change_time=None,
                 max_rows=None, max_samples=None, seed=0, data_dir=None,
                 extrapolation_cls=None, kernel_prior_init=None):
        df = load_labdata_long(data_dir, max_rows=max_rows)
        locs = load_mote_locations(data_dir)
        df = df[df["sensor"].isin(locs)]              # motes without a known location
        if exclude_sensors:
            df = df[~df["sensor"].isin(list(exclude_sensors))]

        wide = pivot_field(df, field, resample=resample, sensors=sensors,
                           time_range=time_range)

        coverage = wide.notna().mean()
        stds = wide.std()
        if sensors is None:
            # a constant column (dead/stuck mote) has full coverage but no information
            ok = coverage[(coverage >= min_coverage) & (stds > 1e-9)]
            if random_selection:
                # random subset instead of best-covered, reproducible through seed.
                rng = np.random.RandomState(seed)
                want = (num_points + N_test) if num_points is not None else len(ok)
                if want > len(ok):
                    print(f"  [warn] requested {want} motes but only {len(ok)} pass "
                          f"min_coverage={min_coverage}; using all of them")
                idx = rng.permutation(len(ok))[:want]
                sensors = [int(ok.index[i]) for i in idx]
            else:
                if num_points is not None:
                    ok = ok.sort_values(ascending=False).head(num_points + N_test)
                sensors = sorted(int(s) for s in ok.index)
        else:
            sensors = [int(s) for s in sensors]
            low = [s for s in sensors if s not in wide.columns
                   or coverage.get(s, 0.0) < min_coverage]
            if low:
                raise ValueError(f"motes {low} are below min_coverage={min_coverage}")
            flat = [s for s in sensors if stds.get(s, 0.0) < 1e-9]
            if flat:
                print(f"  [warn] motes {flat} are constant over the window "
                      f"(dead/stuck?) -- consider dropping them")
        if len(sensors) < 2:
            raise ValueError(f"only {len(sensors)} usable motes "
                             f"(best coverages: {coverage.sort_values(ascending=False).head(10).round(2).to_dict()})")
        wide = wide[sensors]

        Y, time_index = densify(wide.to_numpy(float), wide.index.to_numpy())

        coords = np.array([locs[s] for s in sensors], float)
        coords = coords - coords.mean(axis=0, keepdims=True)   # local frame, better conditioning

        self.field = field
        self.resample = resample
        super().__init__(coords, Y, N_test=N_test, split_time=split_time,
                         change_time=change_time, max_samples=max_samples,
                         seed=seed, time_index=time_index, sensor_ids=sensors,
                         extrapolation_cls=extrapolation_cls,
                         kernel_prior_init=kernel_prior_init)
