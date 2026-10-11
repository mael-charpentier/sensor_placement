"""Where the project keeps its data and its output.

The raw records are not part of the installed package (they are hundreds of megabytes), so
point SPGPRCPD_DATA at the dataset directory:

    export SPGPRCPD_DATA=/scratch/me/spgprcpd-data

Likewise SPGPRCPD_OUTPUT relocates where the results and the logs are written.
"""
import os
from pathlib import Path

#: Root of the source checkout: this file is <root>/src/spgprcpd/paths.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Raw sensor records. Override with SPGPRCPD_DATA.
DATASET_DIR = Path(os.environ.get("SPGPRCPD_DATA", PROJECT_ROOT / "dataset"))

#: Where results and logs are written. Override with SPGPRCPD_OUTPUT.
#: Relative by default, so a run writes beside wherever it was launched.
OUTPUT_DIR = Path(os.environ.get("SPGPRCPD_OUTPUT", "."))

LABDATA_DIR = DATASET_DIR / "labdata"
ECOSENSE_DIR = DATASET_DIR / "ecosense" / "csv_ecopivot"

RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"


def require(path, what):
    """Return path as a string, or raise with a message that says how to fix it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"The raw records are not distributed with the code, see README.md for how to "
            f"obtain them.\n"
            f"If they are already on disk somewhere else, set SPGPRCPD_DATA to the directory "
            f"holding 'labdata/' and 'ecosense/' (currently {DATASET_DIR}).")
    return str(path)
