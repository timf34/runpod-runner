# runpod-runner (`rp`)

A small CLI for provisioning and driving RunPod GPU pods from your laptop, built so that a
Claude Code session can run an experiment end to end:

```
rp up  ->  rp bootstrap  ->  rp run  ->  rp logs  ->  rp down
```

Plain Python 3.11+ stdlib + `requests`. No framework, no daemon, no state anywhere but
`~/.runpod-runner/pods/<name>.json` (one file per pod) and a per-pod `known_hosts`.

## Install

```bash
git clone https://github.com/timf34/runpod-runner && cd runpod-runner
pipx install -e .            # or: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
rp --version
```

(`pip install -e .` straight into Homebrew's Python needs `--break-system-packages`; use pipx
or a venv instead.) The `profiles/` directory is read relative to the package, so an editable
install is the intended mode; set `RP_PROFILES_DIR` if you install it any other way.

Requirements on the laptop: `ssh`/`scp` (OpenSSH), `git`, and `gh` (GitHub CLI, logged in) only
if you use `rp bootstrap --deploy-key`.

## Config

The RunPod API key is read from the environment variable **`RUNPOD_API_KEY`**, or from
**`~/.config/runpod-runner/config.toml`**, which must be yours only (`chmod 600`):

```toml
api_key = "rpa_..."                  # or leave out and export RUNPOD_API_KEY

[defaults]
gpu = "h100"                         # alias (rp gpus) or exact gpuTypeId(s), comma separated
image = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
disk = 60                            # container disk GB (wiped on stop/reset)
volume = "cdv10pb3cq"                # network volume id, or a datacenter like "EUR-IS-3"
cloud = "SECURE"
ssh_key = "~/.ssh/id_ed25519"        # this key's .pub is injected into every pod you create
extra_public_keys = [                # e.g. the runpodctl keys the web UI normally injects
  "ssh-rsa AAAA... runpodctl",
  "~/.ssh/other.pub",
]
git_name  = "Tim Farrelly"           # identity for commits made on the pod (deploy-key flow)
git_email = "you@example.com"

[gpus]                               # extra / overriding aliases
cheap = ["NVIDIA A40", "NVIDIA L40S"]
```

Never commit this file. `rp` warns if its mode is not 600. `RP_HOME` (default `~/.runpod-runner`)
and `RP_CONFIG` override the state dir and config path.

Built-in GPU aliases (`rp gpus`): `h100` -> H100 80GB HBM3, H100 NVL, H100 PCIe (first available
wins); `a100` -> A100-SXM4-80GB, A100 80GB PCIe; `h200` -> H200; plus `a40`, `l40s`, `a6000`, `4090`.
Anything else is passed through verbatim as a gpuTypeId.

## The workflow

| command | what it does |
|---|---|
| `rp up --name N [--gpu h100] [--gpus 1] [--volume EUR-IS-3\|<id>\|none] [--disk 60] [--image ...] [--no-wait] [--dry-run]` | `POST /pods`, poll until ip/port exist, poll until sshd answers, save `~/.runpod-runner/pods/N.json`. Pods using a network volume are pinned to that volume's datacenter. `--dry-run` prints the request body and makes no API call. |
| `rp ls [--json]` / `rp status N [--json]` / `rp volumes [--json]` | account overview ($/h, ip:port, age; running pods highlighted), one pod (also accepts a raw pod id), network volumes. |
| `rp ssh N [-- cmd]` / `rp scp N <src> <dst> [-r]` | interactive shell or a one-off command; copy files either way, the pod side prefixed with `pod:`. |
| `rp bootstrap N --repo URL [--branch b] [--env .env] [--req FILE ...] [--profile hf-latest] [--venv-name X] [--deploy-key]` | on the pod: print GPU + driver CUDA and **fail loudly if the chosen requirements need a newer driver**, install tmux, clone (or `git pull --ff-only` if already there), install `.env` (mode 600), create `/workspace/<repo>_venv`, `pip install -r ...`, then `import torch; assert torch.cuda.is_available()`. Streams to your terminal and logs to `/workspace/bootstrap.log`. Idempotent. |
| `rp run N [--job J] [--cwd DIR] [--env K=V] [--dotenv] -- <command...>` | starts the command in a detached tmux session `J`, in the repo dir with the venv active and `HF_HOME=/workspace/hf PYTHONUNBUFFERED=1`, output to `/workspace/J.log` ending with `EXIT=<code>`. |
| `rp logs N [--job J] [-f] [-n 50]` / `rp jobs N` / `rp kill N --job J` | tail the log (follow with `-f`); list tmux sessions + known jobs with running/EXIT status; kill a job's process group and session. |
| `rp down N [--terminate] [-y]` | stop (default) or delete the pod, delete its GitHub deploy key if one was registered, keep the local state file marked `stopped`/`terminated`. |
| `rp start N` | resume a stopped pod and re-resolve ip/port (new container: apt packages and tmux sessions are gone, `/workspace` survives). |
| `rp adopt <pod-id> [--name N]` / `rp forget N` | import a pod you made in the web UI; drop a local state file (never touches the pod). |

`--` separates rp's options from the remote command; everything after it is joined with spaces and
run by the pod's shell, so `rp ssh N -- "nvidia-smi && df -h /workspace"` works.

### Requirements profiles (`profiles/`)

| profile | contents | needs driver CUDA |
|---|---|---|
| `hf-latest` | torch, transformers>=4.56, accelerate, huggingface_hub, hf_transfer | >= 12.8 |
| `vllm-0.11` | vllm==0.11.0, transformers==4.57.3 | >= 12.8 |
| `vllm-cu124` | torch 2.6+cu124, vllm 0.8.5.post1, transformers 4.51.3 (the coherent old stack from TROUBLESHOOTING) | >= 12.4 |

Each file carries a `# rp-min-cuda: X.Y` header; `rp bootstrap` passes the highest one to
`bootstrap.sh`, which compares it with the host driver's max CUDA (the `nvidia-smi` header, not
`nvcc`) and exits 3 with a clear message if the host is too old. `--req` takes a local file
(uploaded) or a repo-relative path on the pod; with neither `--req` nor `--profile`,
`<repo>/requirements.txt` is installed if it exists.

Pin `torch` in every requirements file (`torch==2.8.*` for driver CUDA 12.8): an unpinned `torch`
resolves to the newest PyPI wheel, built for a newer CUDA than most RunPod hosts' drivers, and the
sanity step then fails with "NVIDIA driver ... too old". The `hf-latest` profile pins it.

### Pushing from the pod without your GitHub token

`rp bootstrap N --repo https://github.com/owner/repo --deploy-key` generates an ed25519 key **on
the pod**, registers it as a *write deploy key* on that repo with `gh api` (the laptop's `gh`
login; the token never leaves the laptop), sets `origin` to the ssh URL with
`core.sshCommand "ssh -i /root/.ssh/deploy_ed25519 -o IdentitiesOnly=yes"`, adds github.com to
known_hosts and sets the git identity. The key id is stored in the state file and **deleted by
`rp down`**. Re-running is a no-op while the same key exists on the pod.

## Example session

```bash
export RUNPOD_API_KEY=...                                  # or config.toml
rp volumes                                                 # pick the volume / datacenter
rp up --name ab-sweep --gpu h100 --volume EUR-IS-3 --disk 60
#   creating pod ab-sweep: gpu=NVIDIA H100 80GB HBM3... x1 image=runpod/pytorch:... volume=cdv10pb3cq (EUR-IS-3)
#   created pod id=abcd1234  $3.29/h
#   waiting for public ip/port ...  ip:port = 213.181.105.231:10406; waiting for sshd ...
#   pod ready: ab-sweep  id=abcd1234  root@213.181.105.231 -p 10406  gpu=NVIDIA H100 80GB HBM3  $3.29/h

rp bootstrap ab-sweep --repo https://github.com/timf34/AttractorBench.git \
    --env ~/secrets/attractorbench.env --profile hf-latest --deploy-key
#   --- GPU / driver ... driver max CUDA: 12.8 ... driver CUDA 12.8 >= required 12.8: ok
#   --- repo ... --- .env installed ... --- venv ... --- sanity torch 2.x cuda available: True
#   === rp bootstrap OK ===

rp run ab-sweep --job sweep1 -- python -u run_suite.py --config configs/frontier.py
rp logs ab-sweep -f                  # Ctrl-C stops following, not the job
rp jobs ab-sweep                     # running / EXIT=0 / EXIT=1 per job
rp scp ab-sweep pod:/workspace/AttractorBench/results ./results -r
rp down ab-sweep                     # stop (container disk wiped, /workspace volume kept); rp start ab-sweep to resume
rp down ab-sweep --terminate -y      # delete the pod for good
```

`rp up --dry-run` prints the exact `POST /pods` body (useful for checking names, env and volume
pinning) and never calls the API.

## Safety notes

- `rp` only touches pods it knows about by **local name**. `rp down` on a pod that was *adopted*
  (not created by `rp up`) or any `--terminate` asks for confirmation, and refuses without `--yes`
  when stdin is not a terminal. **Never stop a pod you did not start without asking its owner.**
- Pod env changes via `PATCH /pods/{id}` reset the container (and were observed to make the pod
  vanish); `rp` never PATCHes. To change a pod, `rp down --terminate` and `rp up` again.
- Stopping a pod wipes the **container disk** (apt packages, `/root`, tmux sessions). Everything
  you care about should live under `/workspace` (repo, venv, `HF_HOME=/workspace/hf`, logs). A
  stopped pod still bills for its disk; `--terminate` stops that.
- Secrets: the API key lives in the environment or a 600 config file; `.env` files are uploaded to
  `/workspace/<repo>/.env` with mode 600; deploy keys are created on the pod and removed on
  `rp down`. Nothing secret goes into this repo or the state files except the Jupyter password
  generated per pod.
- API errors print the HTTP status and response body; ssh failures print the remote exit code.
  Nothing is swallowed.

## Tests

```bash
pip install -e '.[dev]'
pytest                     # fully offline: fake RunPod client + scripted fake ssh/scp; also runs
                           # profiles/bootstrap.sh locally against a fake nvidia-smi
RUNPOD_API_KEY=... pytest -m live    # one read-only GET /pods smoke test
```

## Layout

```
rp/cli.py        argparse + the commands          rp/api.py        RunPodClient (requests)
rp/config.py     key + defaults (env / toml)      rp/state.py      ~/.runpod-runner/pods/<name>.json
rp/ssh.py        the single ssh/scp argv builder  rp/pods.py       polling, volume lookup, helpers
rp/bootstrap.py  bootstrap + deploy keys          rp/jobs.py       tmux job scripts
rp/gpus.py       GPU aliases                      rp/term.py       colours + tables
profiles/        bootstrap.sh + requirements profiles
tests/           pytest (offline), tests/test_live.py (opt-in)
CLAUDE.md        the recipe for Claude Code sessions
```
