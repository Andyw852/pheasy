# AGENTS.md

pheasy: phonon force-constant fitting (2nd/3rd-order IFCs via compressive
sensing). Flat-layout package at the repo root; `core/optimizer.py` holds the
five fitting methods (OLS / LASSO / ALASSO / RFE / RFE-OLS-TSQR) plus legacy
RIDGE. Local editable install runs under the `atomate2_p_a` conda env.

> Identifiers are scrubbed from this public file: the GPU box is `<gpu-host>`,
> its user `<user>`, its pheasy checkout `$REMOTE/pheasy/`. The real values
> (ssh alias, host:port, user, absolute paths) live in `~/.ssh/config` and the
> gitignored `AGENTS.local.md` next to this file.

## Operations (hard-won, non-obvious — read before touching the GPU box / git)

- **Kill GPU jobs by PID, never `pkill -f <script>`.** `pkill -f` matches
  the ssh wrapper's own command line and kills every job whose command contains
  the pattern (it once killed the c7 n=45 holdout while targeting the n-scan).
  Get PIDs with `ps aux | grep <name>` first, then `kill <pid>`.
- **rsync to the GPU box needs explicit absolute paths, one file per call.**
  `~/software/pheasy/...` silently fails to update. Use
  `<gpu-host>:$REMOTE/pheasy/<path>`. With multiple sources, a nested path
  (`core/optimizer.py` plus root files) lands in the wrong directory —
  separate calls.
- **`git push` needs `HOME` set to the local user's home** (the credential
  store lives there) plus `GIT_TERMINAL_PROMPT=0`. GitHub is intermittently
  unreachable (GnuTLS / Empty reply / timeouts): push in a retry loop
  (`for i in 1..8; timeout 120 git push origin master ...`).
- **Check `git branch --show-current` before committing.** A background
  sanitizer (path-scrubbing for the public repo) can leave the worktree on a
  `push-sanitized` branch; commits then land off `master` and
  `git push origin master` silently pushes nothing.
- **The GPU box is shared.** Load is frequently 2x oversubscribed by other
  users' gmx/mdrun jobs; long runs take ~5x wall time. Launch with
  `nohup ... > out 2>&1 &` and poll the output file; never block on them.
- **JS template literals in run_code mangle `\n`** into real newlines when
  writing Python files — use `String.raw` or an array-join so Python string
  literals keep their escapes.

## Data / tools

- c7 (c3=7.0) material + scan results live on the GPU box at
  `$REMOTE/pheasy/MnIn2Se4_c7/`; local c5.2 material at `test/MnIn2Se4/`.
- `holdout_eval.py`: grouped (by-config) cross-method holdout — the direct
  generalization evidence for "does L1 sparsification help vs OLS / RIDGE".
  Runs on the GPU box with the `wc` conda env.
- `scan_methods.sh` / `analyze_scan.py` / `scan8|scan9b` under
  `.scan_results/`: NDATA x method convergence scans and reporting.
