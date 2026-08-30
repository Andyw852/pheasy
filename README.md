# pheasy

Force-constant extraction from finite-displacement / AIMD data.

## Install

```bash
pip install -e .          # exposes the `pheasy` command
pip install -e '.[fast]'  # + celer, a faster LASSO solver
```

## Fitting methods (`-l`)

| flag | method |
|---|---|
| `OLS` | ordinary least squares (LSMR / SVD) |
| `LASSO` | L1 with cross-validated alpha, then debias refit |
| `ALASSO` | adaptive LASSO (Zou 2006) |
| `RFE` | recursive feature elimination, OLS base, grouped CV |
| `RFE-OLS-TSQR` | RFE with a Q-less tall-skinny QR base solver |
| `RIDGE` | L2 with cross-validated alpha |

## Typical workflow

```bash
python3 tools/prepare_dataset.py POSCAR SPOSCAR dataset_disps.npy dataset_forces.npy
pheasy --dim 3 3 3 -w 3 -s --c3 5.2
pheasy --dim 3 3 3 -w 3 -c --c3 5.2
pheasy --dim 3 3 3 -w 3 -d --c3 5.2 --ndata 45 --disp_file
pheasy --dim 3 3 3 -w 3 -f --c3 5.2 --ndata 45 -l OLS --full_ifc --hdf5
```

## Notes

- LASSO / ALASSO need a tight tolerance. `--tol 1e-3` is *not* tight: sklearn
  scales it by `||y||^2`, coordinate descent stops early at small alpha, the CV
  curve goes flat and the fit ends up over-regularized. Use `--tol 1e-6`.
- `PHEASY_SM_DTYPE` (`float64` default) controls the precision of SM / NS / FM.
  All of them must agree, otherwise scipy silently upcasts.
- The inner CV path runs looser than the final refit regardless of `--tol`:
  `PHEASY_CV_TOL` defaults to `max(tol, 1e-3)` and `PHEASY_CV_MAX_ITER` to
  `min(max_iter, 800)`, so CV runs at ~1e-3 even with `--tol 1e-6`.
  `PHEASY_LASSO_DEBIAS` defaults to 1: `results["coef"]` is an OLS refit on the
  selected support, not the raw L1 solution.
