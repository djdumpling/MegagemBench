# Packaged model artifacts

These are the frozen artifacts the published results were produced with. They
ship inside the package so the selector, the interactive demo, and the
reproduction tools all work from a clean install with no download step.
Resolve them in code with `megagem.assets.asset_path(name)`.

| File | Size | What it is | Used by |
|---|---|---|---|
| `ev_dist_v1.pkl` | 572K | Price law F̂ (level 1), fit on teacher games | library default (`EvDistSelector`, piKL `ev_dist` target) |
| `ev_dist_l2_v1.pkl` | 644K | Price law F̂ (level 2), final-ecology refit | the **certified deployable selector** (λ=0.5, δ=2.0, gate=1.0) and interactive play |
| `ev_dist_bp_v1.pkl` | 520K | Price law F̂ (round-1 "blueprint") | reproduction only: dynamics simulator, distillation corpus export |
| `value_head.pkl` | 26K | Supervised gem-value estimator V̂ | selector + piKL search |
| `dynamics_sim_sweep_recovered_200.json` | 17K | Certified (λ, δ, gate) sweep | evidence behind the 0.5/2.0/1.0 config |

## Provenance

The three price laws are distinct fits over **different opponent populations**,
not copies of one model. Each carries its own metadata under the `meta` key
(`pickle.load(...)["meta"]`), including the out-of-fold quality it was accepted
at:

| Artifact | Policy modeled | Fit rows | OOF MAE | OOF bias | within ±1 |
|---|---|---|---|---|---|
| `ev_dist_v1.pkl` | teacher games (E1 frozen) | 3,642 (164 games) | 1.230 | +0.013 | 69.4% |
| `ev_dist_l2_v1.pkl` | `distilled(all-seats, level2)` | 7,020 | 1.279 | +0.026 | 68.8% |
| `ev_dist_bp_v1.pkl` | `blueprint(seat0)` | 1,821 | 1.292 | −0.027 | 66.0% |

SHA-256 (first 16 hex chars), so you can confirm you have the artifacts the
published numbers used:

```
4df84250b64d2604  ev_dist_v1.pkl
2e4fcf11a06d2764  ev_dist_l2_v1.pkl
28282f282c6db732  ev_dist_bp_v1.pkl
9f5c36e3bf811c81  value_head.pkl
a5df609daba64c23  dynamics_sim_sweep_recovered_200.json
```

## Regenerating

| Artifact | Command |
|---|---|
| `ev_dist_*.pkl` | `uv run python scripts/analysis/build_ev_dist_artifact.py` (per-profile; see `--help`) |
| `value_head.pkl` | `uv run python -m megagem.value_head.train --globs '<corpus>/*.json'` |
| sweep JSON | `uv run python scripts/analysis/dynamics_sim.py --mode sweep` |

The regeneration scripts write into `results/` (gitignored scratch). Copy a
regenerated artifact over the packaged one deliberately — don't point defaults
at scratch, and expect the checksums above to change.

## Caveats

- **Pickles execute arbitrary code on load.** Load these only from this
  repository or from artifacts you built yourself. `value_head.pkl` is a plain
  dict of numpy arrays and could be moved to `.npz`; the three price laws are
  scikit-learn `HistGradientBoosting` models, which would need `skops` to store
  in a non-executing format.
- The `.pkl` files were fit with **scikit-learn 1.8.x** and sklearn pickles are
  not stable across minor versions — hence the `scikit-learn~=1.8.0` pin in
  `pyproject.toml`. If you bump sklearn, refit rather than expecting these to
  load.
