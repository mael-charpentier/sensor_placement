# spgprcpd

## Install

Python **3.10** specifically - `tensorflow-probability` 0.25 and `gpflow` 2.10 are pinned to
versions that do not support newer interpreters.

That creates `.venv/` and installs the package:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[milp]"      # gurobipy, the exact MILP for the KL criterion (needs a licence)
pip install -e ".[rds]"       # pyreadr, to read the EcoSense .rds export
```

Without `milp`, the KL criterion falls back to the greedy and exhaustive optimisers. Without
`rds`, the EcoSense loader skips the `.rds` source and uses the curated csv chunks.

---

## Get the data


| Dataset | Kind | Access | Key |
|---|---|---|---|
| Intel Berkeley Lab | indoor mote network, 54 nodes | `dataset/labdata/` (34 MB) | - |
| EcoSense Montreal | outdoor AQ pilot, Parc Jean-Drapeau | `dataset/ecosense/` (527 MB) | - |

---

## How it fits together

```
   ┌─ dataset ──────────────┐
   │  synthetic generator   │   spgprcpd.data
   │  Intel Berkeley Lab    │   one manager interface for all three:
   │  EcoSense Montreal     │   V, dist_prior/dist_post, get_dataset(A, B, phase=...)
   └───────────┬────────────┘
               │  historical record  ->  estimate f^0 (and f^1, synthetic only)
               ▼
   ┌─ placement ────────────┐   spgprcpd.placement
   │  entropy / MI / KL     │   pick A, |A| = b, from the candidates
   │  weighted MI+KL family │   greedy / exhaustive / local search
   └───────────┬────────────┘
               │  A
       ┌───────┴────────┐
       ▼                ▼
┌─ inference ───┐  ┌─ detection ──────────┐
│ exact GP x3   │  │ calibrate θ on f^0   │   spgprcpd.detection
│ (fit every /  │  │ CUSUM / EWMA / BOCPD │   thresholds are never hand-picked:
│  once / never)│  │ SegCPGP / ensembles  │   they are calibrated to a common ARL₀
│ streaming     │  │                      │
└───────┬───────┘  └──────────┬───────────┘
        │ MSE, 95% coverage   │ delay, false-alarm rate, F1
        └──────────┬──────────┘
                   ▼
              results/*.json
```

**The three exact GPs are one model, fitted at three frequencies.** They differ in nothing
else, with the same conditioning set, same predictions given the same kernel:

| Registry name | Class | Kernel fitted |
|---|---|---|
| `exact` | `Exact_GP` | on every batch |
| `exact_no_retrain` | `Exact_GP_no_retraining` | on the first batch only |
| `exact_no_train` | `Exact_GP_no_training` | never - the oracle, handed the true kernel |

`exact` vs `exact_no_retrain` isolates what continuous re-estimation buys; `exact_no_retrain`
vs `exact_no_train` isolates what estimating the kernel at all buys.

---

## Licence and citation

Licensed under **Apache-2.0** - see [`LICENSE`](LICENSE).

Two components are derived from published research code. Both licences are permissive and
**compatible with Apache-2.0**, so the combined work can ship under Apache-2.0:

| Module | Upstream | Licence |
|---|---|---|
| `inference/pep.py`, `inference/sparse.py` | [streaming_sparse_gp](https://github.com/thangbui/streaming_sparse_gp) | Apache-2.0 |
| `detection/segcpgp.py` | [segcpgp](https://github.com/JVerbeek/segcpgp) | MIT |


```bibtex
@mastersthesis{TODO,
  title  = {TODO},
  author = {TODO},
  school = {TODO},
  year   = {TODO},
}
```
