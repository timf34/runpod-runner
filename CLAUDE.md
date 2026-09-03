# CLAUDE.md -- driving RunPod pods with `rp`

`rp` provisions and drives RunPod GPU pods from this laptop. Workflow:
`rp up -> rp bootstrap -> rp run -> rp logs -> rp down`. Everything is a plain CLI call;
exit code 0 = success, 1 = error printed to stderr as `rp: error: ...` (API errors include the
HTTP status + body).

## Setup / where things are

- Install: `pipx install -e .` (or `.venv/bin/pip install -e '.[dev]'`; then use `.venv/bin/rp`).
- API key: env `RUNPOD_API_KEY`, else `~/.config/runpod-runner/config.toml` (mode 600). If neither
  is set, `rp ls` says so. **Never print, echo, or commit the key.**
  To load it for one shell: `export RUNPOD_API_KEY=$(grep '^RUNPOD_API_KEY=' path/to/runpod.env | cut -d= -f2-)`.
- Local state: `~/.runpod-runner/pods/<name>.json` (id, ip, port, repo dir, venv, jobs, deploy
  key id). `rp ls` shows the LOCAL column mapping RunPod pods to these names.
- Pod-side layout: `/workspace` is the network volume (persists). Repo at `/workspace/<repo>`,
  venv at `/workspace/<repo>_venv`, HF cache `/workspace/hf`, job logs `/workspace/<job>.log`,
  bootstrap log `/workspace/bootstrap.log`, job scripts `/workspace/.rp/jobs/<job>.sh`.

## Run an experiment

```bash
rp volumes                                     # volumes are per-datacenter; the pod lands in that DC
rp up --name <name> --gpu h100 --volume EUR-IS-3          # add --dry-run to see the request body first
rp bootstrap <name> --repo <https url> --env <local .env> [--profile hf-latest|vllm-0.11|vllm-cu124] [--deploy-key]
rp run <name> --job <job> -- python -u script.py --args   # detached tmux; log /workspace/<job>.log
rp logs <name> --job <job> -n 100                         # or -f to follow (Ctrl-C detaches only)
rp jobs <name>                                            # running / EXIT=<code> per job
rp scp <name> pod:/workspace/<repo>/results ./results -r
rp down <name>                                            # stop; rp start <name> resumes; --terminate deletes
```

- `--` is mandatory before the remote command for `rp run` / `rp ssh`; the words after it are joined
  with spaces and run by the pod's shell (pipes, `&&`, env assignments all work).
- `rp run` runs in the bootstrapped repo dir with the venv active and exports
  `HF_HOME=/workspace/hf PYTHONUNBUFFERED=1`. Add `--env K=V` for more, `--dotenv` to
  `set -a; source .env` first (not needed when the code calls `load_dotenv()` itself).
- Always `python -u` (unbuffered) so `rp logs` shows progress.
- A job's log ends with `EXIT=<code>` when it finishes (`EXIT=KILLED` after `rp kill`). Poll with
  `rp jobs <name>` or `rp logs <name> -n 5`; do not sit in `rp logs -f` for hours.

## Babysitting a long job

- `rp jobs <name>` every few minutes is the cheap check (one ssh round trip).
- `rp logs <name> -n 30` for the tail; `rp ssh <name> -- "grep -c 'done' /workspace/<job>.log"` for
  ad-hoc greps. Wrap greps so a missing file does not look like a crash.
- ssh hiccups (exit 255) are transient; retry once before concluding the pod died. `rp status <name>`
  refreshes ip/port from the API (pods get a new ip/port after `rp start`).
- If the pod is EXITED/stopped but you did not stop it, say so; do not recreate without asking.

## Gotchas (learned the hard way)

- **Driver CUDA**: the host driver's max CUDA (`nvidia-smi` header, *not* `nvcc`) must be >= the
  torch/vLLM wheel's CUDA. `rp bootstrap` prints it and fails with exit 3 if the chosen profile needs
  more; on a 12.4-driver host use `--profile vllm-cu124`. Never pip-upgrade packages into the image's
  system Python; use the venv `rp bootstrap` makes.
- **Unpinned `torch` pulls a wheel newer than the driver** (2026-09-03, AttractorStatePrefillAttack): a
  requirements file with a bare `torch` line resolved to torch 2.14 (cu130) on a driver-12.8 host, and the
  sanity step failed with "NVIDIA driver ... too old (found version 12080)", cuda available: False. Always
  pin torch to a line built for the host driver (`torch==2.8.*` for CUDA 12.8; the `hf-latest` profile now
  does). The bootstrap error message says this too.
- **pkill over ssh kills your own ssh shell** if the pattern matches the command line you are running.
  Use `pkill -f "[p]attern"` or better `rp kill <name> --job <job>` (kills the process group + tmux).
- **tmux session names** cannot contain `.` or `:`; `rp` validates `--job` accordingly.
- **Container disk is wiped** on stop/reset: apt packages (incl. tmux), `/root/.ssh/deploy_ed25519`,
  running tmux sessions all vanish. `/workspace` survives. `rp bootstrap` is idempotent -- re-run it
  after `rp start` (it reinstalls tmux, pulls, reuses the venv, re-registers a deploy key if asked).
- **Never PATCH a pod** (env change => container reset, pod observed to vanish). `rp` does not;
  terminate + `rp up` instead.
- `git clone` of a public repo works over https; pushing from the pod uses the deploy-key flow
  (`--deploy-key`), which never copies your GitHub token anywhere. `rp down` deletes that key.
- `system pip` on the image needs `--break-system-packages`; the venv avoids the question.

## Hard rules

- **Never stop, restart, reset, patch, delete or ssh into a pod you did not create in this session
  without asking the user first.** Other pods on the account may be running someone's expensive job.
  `rp down` refuses adopted pods and `--terminate` without `--yes` when non-interactive -- do not add
  `--yes` to get around a prompt unless the user told you to.
- Never create a pod "to test"; every pod costs money from the moment `POST /pods` returns.
  Use `rp up --dry-run` to check the request and the offline test suite (`pytest`) for logic.
- Never write the API key, `.env` contents, or deploy keys into files under version control.
- When done, `rp down <name>` (stop) unless the user wants the pod kept; say which pods are still
  running and their $/h (`rp ls` prints the total).
