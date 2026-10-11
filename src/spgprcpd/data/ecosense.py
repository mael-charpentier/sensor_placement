"""Data manager for the real EcoSense Montreal deployment (dataset/ecosense/csv_ecopivot).

The raw log is a long csv (one row per reading, per sensor) with irregular per-sensor
timestamps, mobile gps coordinates (many NA) and ~16 measurement channels.  The loader

  1. Reads one of the curated CSVs (or every ecosense_*.csv chunk),
  2. Resamples each sensor onto a common regular grid (mean per bin, default 15 min),
  3. Keeps the requested sensors (or the best-covered ones),
  4. Converts the per-sensor median gps fix (DDMM.MMMM) to a local metric frame,

and hands the dense (T, n) block to spgprcpd.data.real_base.Real_Data_Manager_Base,
which exposes the exact interface of the synthetic manager (V, X/y/y_iid, empirical
dist_prior/dist_post, historical/test phases).

Example:
    from spgprcpd.data.ecosense import Ecosense_Data_Manager
    dm = Ecosense_Data_Manager(field="Temperature (C)", source="all",
                               placement_sensors=[9, 17, 18, 19],
                               evaluation_sensors=[7, 18],
                               historical_sensors=[9, 17, 19, 20],
                               time_range=("2025-06-05", None), resample="15min")
    print(dm.describe())
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from spgprcpd import paths
from spgprcpd.data.real_base import Real_Data_Manager_Base, lonlat_to_meters, remove_flat_runs

#: Resolved centrally (see spgprcpd.paths). Override with the SPGPRCPD_DATA env var.
DATA_DIR = str(paths.ECOSENSE_DIR)

# named sources -> file in DATA_DIR. "all" reads every source (all CSVs + the .rds), merged
# and de-duplicated, and cached (see load_ecosense_all).
SOURCES = {
    "change_filtered": "change_filtered_output.csv",
    "filtered": "filtered_output.csv",
    "filtered_plus": "filtered_output_plus.csv",
    "consolidated": "ecosense.csv",
    "all": None,
}

# cache for the merged "all" table (built once by load_ecosense_all, reloaded afterwards)
ALL_CACHE = os.path.join(DATA_DIR, "_ecosense_all.pkl")

# the measurement channels of the log (after whitespace-stripping the header)
FIELDS = [
    "Temperature (C)", "Humidity (%)", "Pressure (mb)", "Light (Lux)",
    "CO (Ohm)", "NO2 (Ohms)", "NH3 (Ohm)",
    "PM1 (ug/m3)", "PM2.5 (ug/m3)", "PM4 (ug/m3)", "PM10 (ug/m3)",
    "PM0.5 (#/cm3)", "PM1.0 (#/cm3)", "PM2.5 (#/cm3)", "PM4.0 (#/cm3)", "PM10.0 (#/cm3)",
    "Noise (amplitude)",
]


def _to_decimal(coord, direction):
    """DDMM.MMMM (+ N/S/E/W letter) -> decimal degrees."""
    if pd.isna(coord) or coord == "":
        return np.nan
    if isinstance(coord, str):
        cleaned = re.findall(r"[0-9.]+", coord)
        if not cleaned:
            return np.nan
        coord = float(cleaned[0])
    decimal = coord // 100 + (coord % 100) / 60
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal


def _dedup_columns(df):
    """Strip whitespace from column names and merge duplicate-named columns (a channel may
    appear as both 'Temperature (C)' and ' Temperature (C)') by taking the first non-null,
    so df[name] is always a single Series afterwards."""
    df = df.rename(columns=lambda c: str(c).strip())
    if not df.columns.duplicated().any():
        return df
    out = {}
    for name in pd.unique(df.columns):
        sub = df.loc[:, df.columns == name]
        col = sub.iloc[:, 0]
        for k in range(1, sub.shape[1]):
            col = col.combine_first(sub.iloc[:, k])
        out[name] = col
    return pd.DataFrame(out)


def _parse_time(s):
    """Timestamps arrive as mixed strings (ISO-Z and naive) in the CSVs, or as datetimes in
    the .rds. Normalise either to naive (tz-stripped) datetimes."""
    if pd.api.types.is_datetime64_any_dtype(s):
        t = pd.to_datetime(s, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(s, format="mixed", utc=True, errors="coerce")
    return t.dt.tz_localize(None)


def _normalize_frame(raw):
    """Turn one raw source (a csv or the .rds) into the long format: columns time
    (datetime64), sensor (int), the channels of FIELDS, Latitude /
    Longitude (decimal degrees, nan when no fix).  Returns None for a file that is not an
    EcoSense reading table (no Date/Time / EcoSense id), so foreign CSVs are skipped."""
    df = _dedup_columns(raw)
    if "Date/Time" not in df.columns or "EcoSense ID" not in df.columns:
        return None
    df["time"] = _parse_time(df["Date/Time"])
    df["sensor"] = pd.to_numeric(df["EcoSense ID"], errors="coerce")
    df = df.dropna(subset=["time", "sensor"])
    if df.empty:
        return None
    df["sensor"] = df["sensor"].astype(int)

    for f in FIELDS:
        if f not in df.columns:
            df[f] = np.nan
        df[f] = pd.to_numeric(df[f], errors="coerce")
        df.loc[df[f] == -1.0, f] = np.nan            # -1 is the sensor-off sentinel

    if {"Latitude (DDMM.MMMM)", "Latitude E (N/S)"} <= set(df.columns):
        df["Latitude"] = [_to_decimal(c, d) for c, d in
                          zip(df["Latitude (DDMM.MMMM)"], df["Latitude E (N/S)"])]
    else:
        df["Latitude"] = np.nan
    if {"Longitude (DDMM.MMMM)", "Longitude E (W/E)"} <= set(df.columns):
        df["Longitude"] = [_to_decimal(c, d) for c, d in
                           zip(df["Longitude (DDMM.MMMM)"], df["Longitude E (W/E)"])]
    else:
        df["Longitude"] = np.nan

    return df[["time", "sensor"] + FIELDS + ["Latitude", "Longitude"]]


def _find_rds(data_dir):
    """Locate ecosense.rds next to the CSVs or in a parent dataset folder (as read by
    test_data.py).  Returns the path or None."""
    for cand in (os.path.join(data_dir, "ecosense.rds"),
                 os.path.join(os.path.dirname(data_dir), "ecosense.rds")):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(os.path.dirname(data_dir), "**", "*.rds"), recursive=True)
    return hits[0] if hits else None


def _read_rds(path):
    """Read the .rds via pyreadr (a single unnamed object). None if pyreadr is unavailable."""
    try:
        import pyreadr
    except ImportError:
        print("  [info] pyreadr not installed -> skipping the .rds source")
        return None
    res = pyreadr.read_r(path)
    key = None if None in res else next(iter(res), None)
    return None if key is None and None not in res else res[key]


def load_ecosense_all(data_dir=None, rebuild=False, cache_path=None):
    """Read every EcoSense source once and cache the result.

    Reads every *.csv in data_dir (the numbered chunks ecosense_<i>.csv, the
    consolidated ecosense.csv, the curated _output.csv slices) and the .rds if
    present (and pyreadr is installed), normalises each (whitespace / duplicate column names,
    the -1 sensor-off sentinel, the DDMM.MMMM coordinates), then de-duplicates readings: the
    same (sensor, time) recorded in several files is collapsed to one row, taking the first
    non-null value of every channel (so partial copies are merged, not lost).

    The (slow, ~1 GB) read is done once: the merged long table is pickled to cache_path
    (default data_dir/_ecosense_all.pkl) and reloaded on later calls.  Pass
    rebuild=True after adding/updating source files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    cache = cache_path or (ALL_CACHE if data_dir == DATA_DIR
                           else os.path.join(data_dir, "_ecosense_all.pkl"))
    if os.path.exists(cache) and not rebuild:
        return pd.read_pickle(cache)

    frames = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        try:
            raw = pd.read_csv(p, na_values=["NA"], low_memory=False)
        except Exception as e:                       # a malformed / foreign csv must not stop the merge
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        norm = _normalize_frame(raw)
        if norm is not None and len(norm):
            frames.append(norm)

    rds = _find_rds(data_dir)
    if rds is not None:
        raw = _read_rds(rds)
        if raw is not None:
            norm = _normalize_frame(raw)
            if norm is not None and len(norm):
                frames.append(norm)

    if not frames:
        raise RuntimeError(f"no EcoSense reading table found under {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    # collapse duplicate readings: groupby.first keeps the first non-null of each column per
    # (sensor, time), so identical rows are merged and partial copies fill each other's gaps.
    df = (df.groupby(["sensor", "time"], as_index=False, sort=False).first()
            .sort_values(["sensor", "time"]).reset_index(drop=True))
    print(f"  EcoSense 'all': {n_raw} rows from {len(frames)} sources "
          f"-> {len(df)} unique readings; cached to {cache}")
    df.to_pickle(cache)
    return df


def load_ecosense_long(source="change_filtered", data_dir=None):
    """Long-format EcoSense record: one row per reading, columns time (datetime64),
    sensor (int), the measurement channels of FIELDS, Latitude/Longitude
    (decimal degrees, nan when no gps fix).  source: a key of SOURCES or a path.    "all" reads and caches every source (see load_ecosense_all)."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if source == "all":
        return load_ecosense_all(data_dir)
    if source in SOURCES:
        paths = [os.path.join(data_dir, SOURCES[source])]
    else:
        paths = [source if os.path.isabs(source) else os.path.join(data_dir, source)]

    frames = []
    for p in paths:
        norm = _normalize_frame(pd.read_csv(p, na_values=["NA"], low_memory=False))
        if norm is not None and len(norm):
            frames.append(norm)
    if not frames:
        raise RuntimeError(f"no usable EcoSense data in {paths}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["sensor", "time"]).reset_index(drop=True)


def pivot_field(df, field, resample="15min", sensors=None, time_range=None):
    """Wide (T_bins, n_sensors) DataFrame of one channel: mean per resample bin per
    sensor, on the common regular grid spanning the (cropped) record.  nan where a sensor has
    no reading in a bin."""
    if field not in df.columns:
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
    # complete regular grid (resample only covers each sensor's own span)
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=resample)
    return wide.reindex(full)


def sensor_coords(df, sensors):
    """(len(sensors), 2) local-metre coordinates from each sensor's median gps fix.
    Raises with the list of offenders if a requested sensor has no gps fix at all."""
    lat, lon, missing = [], [], []
    for s in sensors:
        fixes = df[(df["sensor"] == s) & df["Latitude"].notna() & df["Longitude"].notna()]
        if fixes.empty:
            missing.append(s)
            lat.append(np.nan)
            lon.append(np.nan)
        else:
            lat.append(float(fixes["Latitude"].median()))
            lon.append(float(fixes["Longitude"].median()))
    if missing:
        raise ValueError(
            f"sensors {missing} have no GPS fix in this source; either drop them, pass "
            f"coords_override={{id: (lat, lon)}}, or load a source that contains their fixes")
    return lonlat_to_meters(lat, lon)


class Ecosense_Data_Manager(Real_Data_Manager_Base):
    """Real-data manager for the EcoSense deployment (module docstring).

    Sensor roles, three lists. A sensor may appear in several of them:

    placement_sensors: can be selected by the placement criteria (the candidate set V).
    evaluation_sensors: inference is evaluated there (gaussian process reconstruction, masked to
                         actually-observed entries).  A sensor that is also a placement
                         candidate is evaluated whenever it is not placed.
    historical_sensors: their record feeds the empirical f0 / f1 estimation.  A sensor
                         absent from this list contributes no historical knowledge (its
                         mean/covariance entries fall back to 0, ridge-protected), even if
                         it is placeable or evaluated.

    e.g. Placement=[9,17,18,19], evaluation=[7,9,17,18,19], historical=[9,17,19,20]:
    9/17/19 are classic candidates, 18 is placeable + evaluated but untrusted historically,
    7 is evaluation-only, 20 is historical-only.

    Parameters:
    field: str
        Measurement channel, see FIELDS.
    source: str
        A key of SOURCES, or a path to a csv.
    placement_sensors: list or None
        None falls back to the best-covered sensors, the auto-selection, and those then
        default to being the historical ones too.
    evaluation_sensors: list or None
        None means that, under the auto-selection, the last N_test auto-selected sensors
        become evaluation-only.
    historical_sensors: list or None
        None means every used sensor, which is the classic behaviour.
    num_points: int or None
        Cap on the auto-selection, which keeps num_points + N_test sensors.
    N_test: int
        Only used on the auto-selection path, see above.
    time_range: (start, end) or None
        Crop, as timestamps or strings. An open side is written as None.
    resample: str
        Bin width of the common grid, a pandas offset such as "15min".
    min_coverage: float
        Threshold for the auto-selection. A sensor listed explicitly only gets a warning
        when it falls below it.
    drop_flat_run: int or None
        Turn a run of this many consecutive identical bins into nan, which is how a dead or
        stuck sensor shows up. None disables the check.
    coords_override: dict or None
        A {sensor_id: (lat, lon)} mapping, for the sensors with no gps fix.
    split_time: int or timestamp, or None
        Boundary between the historical and the test phase, half of T by default. The whole
        historical phase is f0, since real data never exposes f1, so dist_post is None.
    change_time: int or timestamp, or None
        A real change annotated in the test stream. None, the default, assumes no change,
        and the detectors are then scored on staying quiet.
    """

    def __init__(self, field="Temperature (C)", source="change_filtered", *,
                 placement_sensors=None, evaluation_sensors=None, historical_sensors=None,
                 num_points=None, N_test=0, time_range=None,
                 resample="15min", min_coverage=0.3, drop_flat_run=8,
                 coords_override=None, split_time=None, change_time=None,
                 max_samples=None, seed=0, data_dir=None,
                 extrapolation_cls=None, kernel_prior_init=None):
        evaluation_sensors = [int(s) for s in (evaluation_sensors or [])]
        historical_sensors = ([int(s) for s in historical_sensors]
                              if historical_sensors is not None else None)

        df = load_ecosense_long(source, data_dir)
        want = None
        if placement_sensors is not None:
            want = sorted(set(int(s) for s in placement_sensors)
                          | set(evaluation_sensors) | set(historical_sensors or []))
        wide = pivot_field(df, field, resample=resample, sensors=want,
                           time_range=time_range)

        # dead/stuck sensors repeat their last reading -> a flat line. Treat it as missing
        if drop_flat_run:
            wide.iloc[:, :] = remove_flat_runs(wide.to_numpy(float), drop_flat_run)

        coverage = wide.notna().mean()
        stds = wide.std()
        if placement_sensors is None:
            # auto-selection: best covered, non-constant, not reserved for another role
            reserved = set(evaluation_sensors) | set(historical_sensors or [])
            ok = coverage[(coverage >= min_coverage) & (stds > 1e-9)]
            ok = ok[~ok.index.isin(reserved)]
            if num_points is not None:
                ok = ok.sort_values(ascending=False).head(num_points + N_test)
            auto = sorted(int(s) for s in ok.index)
            if evaluation_sensors:
                placement_sensors = auto
            else:               # legacy: the last N_test auto-selected are evaluation-only
                placement_sensors = auto[:len(auto) - N_test] if N_test else auto
                evaluation_sensors = auto[len(auto) - N_test:] if N_test else []
        else:
            placement_sensors = [int(s) for s in placement_sensors]
            low = [s for s in placement_sensors
                   if s in wide.columns and coverage.get(s, 0.0) < min_coverage]
            if low:
                # explicitly requested -> keep them, but say so (the record is trimmed to
                # the common span of the placement/evaluation sensors below)
                print(f"  [warn] sensors {low} are below min_coverage={min_coverage} "
                      f"(coverage: {coverage.round(3).to_dict()})")
            flat = [s for s in placement_sensors if stds.get(s, 1.0) < 1e-9]
            if flat:
                print(f"  [warn] sensors {flat} are constant over the window "
                      f"(dead/stuck?) -- consider dropping them")

        if historical_sensors is None:      # classic behaviour: everything is historical
            historical_sensors = sorted(set(placement_sensors) | set(evaluation_sensors))
        if not historical_sensors:
            raise ValueError("historical_sensors is empty: f^0 / f^1 cannot be estimated")
        if len(placement_sensors) < 2:
            raise ValueError(f"only {len(placement_sensors)} placement sensors "
                             f"(coverage: {coverage.round(2).to_dict()})")

        # column layout expected by the base: [placement | evaluation-only | historical-only]
        eval_only = [s for s in evaluation_sensors if s not in placement_sensors]
        hist_only = [s for s in historical_sensors
                     if s not in placement_sensors and s not in eval_only]
        cols = placement_sensors + eval_only + hist_only

        gone = [s for s in cols if s not in wide.columns or wide[s].notna().sum() < 2]
        if gone:
            raise ValueError(f"sensors {gone} have (almost) no data for {field!r} in "
                             f"this window (coverage: {coverage.round(3).to_dict()})")
        wide = wide[cols]
        observed = wide.notna()

        # no interpolation:
        keep = observed.any(axis=1)
        wide, observed = wide.loc[keep], observed.loc[keep]
        n_obs_bins = int(observed[placement_sensors + eval_only].any(axis=1).sum())
        if n_obs_bins < 20:
            raise ValueError(f"only {n_obs_bins} time bins carry a placement/evaluation "
                             f"observation for {cols}; widen time_range or drop the "
                             f"sparsest sensor")
        Y = wide.to_numpy(float)
        time_index = wide.index.to_numpy()

        if coords_override:
            for s, (la, lo) in coords_override.items():
                mask = df["sensor"] == s
                df.loc[mask, "Latitude"] = df.loc[mask, "Latitude"].fillna(la)
                df.loc[mask, "Longitude"] = df.loc[mask, "Longitude"].fillna(lo)
        coords = sensor_coords(df, cols)

        no_hist = [i for i, s in enumerate(cols) if s not in historical_sensors]
        if any(cols[i] in placement_sensors for i in no_hist):
            print(f"  [info] placement sensors "
                  f"{[cols[i] for i in no_hist if cols[i] in placement_sensors]} carry no "
                  f"historical knowledge: the informed criteria have nothing to rank them by")

        self.field = field
        self.source = source
        self.resample = resample
        super().__init__(coords, Y,
                         N_test=len(eval_only), N_hist_only=len(hist_only),
                         observed=observed.to_numpy(bool), no_hist_cols=no_hist,
                         split_time=split_time, change_time=change_time,
                         max_samples=max_samples, seed=seed,
                         time_index=time_index, sensor_ids=cols,
                         extrapolation_cls=extrapolation_cls,
                         kernel_prior_init=kernel_prior_init)
        # placement sensors that must also be evaluated (whenever they are not placed)
        self.eval_extra = [cols.index(s) for s in evaluation_sensors
                           if s in placement_sensors]
        if self.eval_extra:
            # get_dataset: evaluate on (passed B) + the evaluation-only columns
            self.only_testing_loc = 2
