#!/usr/bin/env bash
# rp bootstrap -- uploaded to /workspace/.rp/bootstrap.sh and executed over ssh by `rp bootstrap`.
# Everything is logged to /workspace/bootstrap.log. Idempotent: re-running pulls instead of cloning
# and reuses the venv.
#
# Inputs (environment variables, set by rp):
#   RP_REPO_URL      clone URL (https://... or git@github.com:owner/repo.git)
#   RP_REPO_DIR      e.g. /workspace/AttractorBench
#   RP_BRANCH        optional branch to check out
#   RP_VENV          e.g. /workspace/AttractorBench_venv
#   RP_REQS          space-separated requirement files that MUST exist (uploaded profiles / local files)
#   RP_REQS_OPT      space-separated requirement files that are installed only if present (repo's requirements.txt)
#   RP_DOTENV        path of an uploaded .env to move into the repo root (optional)
#   RP_MIN_CUDA      e.g. 12.8 -> fail loudly if the host driver's max CUDA is older (optional)
#   RP_GIT_SSH_COMMAND  e.g. "ssh -i /root/.ssh/deploy_ed25519 -o IdentitiesOnly=yes" (deploy-key flow, optional)
#   RP_GIT_NAME / RP_GIT_EMAIL   git identity for commits made on the pod (optional)
#   RP_SKIP_SANITY   set to 1 to skip the torch/CUDA import check
#   RP_WORKSPACE / RP_BASHRC / RP_PIP_UPGRADE   test knobs (default /workspace, /root/.bashrc, 1)
#
# Exit codes: 0 ok, 3 driver/GPU problem, 4 git problem, 5 pip problem, 6 CUDA sanity check failed.

set -uo pipefail
WS="${RP_WORKSPACE:-/workspace}"
LOG="$WS/bootstrap.log"
mkdir -p "$WS/.rp"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "--- $*"; }
die()  { echo; echo "!!! rp bootstrap FAILED: $1"; exit "${2:-1}"; }

echo "=== rp bootstrap $(date -u +%FT%TZ) on $(hostname) ==="
echo "repo=$RP_REPO_URL dir=$RP_REPO_DIR branch=${RP_BRANCH:-<default>} venv=$RP_VENV"

step "host"
echo "cpus=$(nproc 2>/dev/null || echo ?)  ram=$(free -g 2>/dev/null | awk '/Mem:/{print $2}')G  $WS: $(df -h "$WS" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"

step "GPU / driver (nvidia-smi header is the truth, not nvcc)"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- is this a GPU pod?" 3
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || die "nvidia-smi failed -- GPU not visible in this container" 3
DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "driver max CUDA: ${DRIVER_CUDA:-unknown}"
if [ -n "${RP_MIN_CUDA:-}" ] && [ -n "${DRIVER_CUDA:-}" ]; then
  LOWEST=$(printf '%s\n%s\n' "$RP_MIN_CUDA" "$DRIVER_CUDA" | sort -V | head -1)
  if [ "$LOWEST" != "$RP_MIN_CUDA" ]; then
    die "the chosen requirements need CUDA >= $RP_MIN_CUDA but this host's driver only supports CUDA $DRIVER_CUDA.
    Pick an older pinned stack (e.g. a cu124 profile) or create the pod on a newer host; see TROUBLESHOOTING." 3
  fi
  echo "driver CUDA $DRIVER_CUDA >= required $RP_MIN_CUDA: ok"
fi

step "tmux"
if ! command -v tmux >/dev/null 2>&1; then
  (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1 && echo "installed tmux" || echo "warn: could not apt-get install tmux (rp run needs it)"
else
  echo "tmux $(tmux -V 2>/dev/null | awk '{print $2}') present"
fi

step "repo: $RP_REPO_URL -> $RP_REPO_DIR"
if [ -n "${RP_GIT_SSH_COMMAND:-}" ]; then export GIT_SSH_COMMAND="$RP_GIT_SSH_COMMAND"; fi
if [ -d "$RP_REPO_DIR/.git" ]; then
  echo "already cloned; git pull --ff-only"
  if [ -n "${RP_BRANCH:-}" ]; then
    git -C "$RP_REPO_DIR" fetch origin "$RP_BRANCH" >/dev/null 2>&1 || true
    git -C "$RP_REPO_DIR" checkout "$RP_BRANCH" || die "could not check out branch $RP_BRANCH" 4
  fi
  git -C "$RP_REPO_DIR" pull --ff-only || die "git pull --ff-only failed (local changes or diverged history in $RP_REPO_DIR)" 4
else
  mkdir -p "$(dirname "$RP_REPO_DIR")"
  if [ -n "${RP_BRANCH:-}" ]; then
    git clone --branch "$RP_BRANCH" "$RP_REPO_URL" "$RP_REPO_DIR" || die "git clone failed" 4
  else
    git clone "$RP_REPO_URL" "$RP_REPO_DIR" || die "git clone failed" 4
  fi
fi
if [ -n "${RP_GIT_SSH_COMMAND:-}" ]; then git -C "$RP_REPO_DIR" config core.sshCommand "$RP_GIT_SSH_COMMAND"; fi
if [ -n "${RP_GIT_NAME:-}" ];  then git -C "$RP_REPO_DIR" config user.name  "$RP_GIT_NAME";  fi
if [ -n "${RP_GIT_EMAIL:-}" ]; then git -C "$RP_REPO_DIR" config user.email "$RP_GIT_EMAIL"; fi
echo "HEAD: $(git -C "$RP_REPO_DIR" log -1 --oneline)"

step ".env"
if [ -n "${RP_DOTENV:-}" ] && [ -f "$RP_DOTENV" ]; then
  mv -f "$RP_DOTENV" "$RP_REPO_DIR/.env" && chmod 600 "$RP_REPO_DIR/.env"
  echo "installed $RP_REPO_DIR/.env ($(wc -l < "$RP_REPO_DIR/.env" | tr -d ' ') lines, mode 600)"
elif [ -f "$RP_REPO_DIR/.env" ]; then
  echo "keeping existing $RP_REPO_DIR/.env"
else
  echo "no .env (pass --env path/to/.env to rp bootstrap if the repo needs one)"
fi

step "venv: $RP_VENV"
if [ -x "$RP_VENV/bin/python" ]; then
  echo "reusing existing venv"
else
  python3 -m venv "$RP_VENV" || die "python3 -m venv failed" 5
fi
# shellcheck disable=SC1090
source "$RP_VENV/bin/activate"
if [ "${RP_PIP_UPGRADE:-1}" = "1" ]; then
  python -m pip install -q -U pip || die "pip self-upgrade failed" 5
fi
echo "python $(python --version 2>&1 | awk '{print $2}') pip $(python -m pip --version | awk '{print $2}')"

for r in ${RP_REQS:-}; do
  step "pip install -r $r"
  [ -f "$r" ] || die "requirements file $r not found" 5
  python -m pip install -r "$r" || die "pip install -r $r failed (driver CUDA is $DRIVER_CUDA -- check wheel/CUDA compatibility)" 5
done
for r in ${RP_REQS_OPT:-}; do
  if [ -f "$r" ]; then
    step "pip install -r $r"
    python -m pip install -r "$r" || die "pip install -r $r failed (driver CUDA is $DRIVER_CUDA -- check wheel/CUDA compatibility)" 5
  else
    echo "(no $r -- skipping)"
  fi
done

step "sanity"
if [ "${RP_SKIP_SANITY:-0}" = "1" ]; then
  echo "skipped (RP_SKIP_SANITY=1)"
else
  python - <<'PY' || die "CUDA sanity check failed (torch cannot see the GPU). If the line above says torch was built for a CUDA newer than the driver max CUDA printed earlier, the requirements pulled a too-new wheel: pin torch to a release built for the driver (e.g. torch==2.8.* for CUDA 12.8; unpinned torch on PyPI now resolves to cu130 builds) and re-run bootstrap" 6
import importlib.util, sys
if importlib.util.find_spec("torch") is None:
    print("torch not installed -- skipping CUDA check"); sys.exit(0)
import torch
print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda}); cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("gpu:", torch.cuda.get_device_name(0))
PY
fi

mkdir -p "$WS/hf"
BASHRC="${RP_BASHRC:-/root/.bashrc}"
grep -q "HF_HOME=$WS/hf" "$BASHRC" 2>/dev/null || echo "export HF_HOME=$WS/hf" >> "$BASHRC"
echo
echo "=== rp bootstrap OK $(date -u +%FT%TZ) ==="
echo "activate: source $RP_VENV/bin/activate   repo: $RP_REPO_DIR   log: $LOG"
