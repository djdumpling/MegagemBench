#!/usr/bin/env bash
# Durable form of the RL plan §5 #6 option (a): fork huggingface/trl at the
# pinned commit, force `is_vllm_available()` -> False so TRL skips its
# vLLM-0.12+-only imports (we never use TRL's internal vLLM path — rollouts go
# through our own external vLLM 0.10.2 server via the rollout_func seam), push
# the patched branch under your GitHub account, and pin pyproject.toml to the
# resulting exact fork SHA.
#
# Idempotent: safe to re-run. Re-running re-applies the patch onto a fresh
# checkout of the pinned commit and force-updates the branch.
#
# Prereqs:
#   - `gh` CLI installed and authenticated (`gh auth status` must pass).
#   - git identity configured.
#   - Run from the MegagemBench repo root.
#
# Usage:
#   bash scripts/training/setup_trl_fork.sh
#   UPSTREAM_SHA=<other> BRANCH=<name> bash scripts/training/setup_trl_fork.sh
#
# After it finishes, install the fork into the venv:
#   uv sync --extra rl          (or: uv pip install -e ".[rl]" --no-deps + deps)

set -euo pipefail

: "${UPSTREAM_REPO:=huggingface/trl}"
: "${UPSTREAM_SHA:=5da60783af85a180f10235e564f37e4da67cc01d}"
: "${BRANCH:=megagem-vllm0102-compat}"
: "${WORKDIR:=$(mktemp -d)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYPROJECT="${REPO_ROOT}/pyproject.toml"

log() { printf '[trl-fork] %s\n' "$*"; }
cleanup() { [[ -n "${WORKDIR:-}" && -d "${WORKDIR}" ]] && rm -rf "${WORKDIR}"; }
trap cleanup EXIT

command -v gh  >/dev/null || { log "FATAL: gh CLI not found."; exit 2; }
command -v git >/dev/null || { log "FATAL: git not found."; exit 2; }
gh auth status >/dev/null 2>&1 || { log "FATAL: gh not authenticated (run: gh auth login)."; exit 2; }

GH_USER="$(gh api user --jq .login)"
[[ -n "${GH_USER}" ]] || { log "FATAL: could not resolve GitHub login."; exit 2; }
log "GitHub user: ${GH_USER}"

# 1. Ensure the fork exists (idempotent — no-op if already forked).
log "ensuring fork ${GH_USER}/trl exists"
gh repo fork "${UPSTREAM_REPO}" --clone=false --remote=false >/dev/null 2>&1 || true

# 2. Check out the upstream pinned commit.
log "fetching upstream ${UPSTREAM_REPO}@${UPSTREAM_SHA:0:12}"
git clone --quiet --filter=blob:none "https://github.com/${UPSTREAM_REPO}.git" "${WORKDIR}/trl"
cd "${WORKDIR}/trl"
git fetch --quiet origin "${UPSTREAM_SHA}"
git checkout --quiet -B "${BRANCH}" "${UPSTREAM_SHA}"

# 3. Patch: append a final `is_vllm_available` def. Last definition at module
#    scope wins regardless of the original body / decorators / line numbers —
#    same mechanism as the previously verified venv hotfix.
IMPORT_UTILS="trl/import_utils.py"
[[ -f "${IMPORT_UTILS}" ]] || { log "FATAL: ${IMPORT_UTILS} not at this commit."; exit 1; }
if grep -q "MEGAGEM-VLLM0102-COMPAT" "${IMPORT_UTILS}"; then
    log "patch marker already present (unexpected on a fresh checkout) — continuing"
else
    cat >> "${IMPORT_UTILS}" <<'PATCH'


# --- MEGAGEM-VLLM0102-COMPAT (the RL plan §5 #6) ---------------------------
# We pin vLLM 0.10.2 (CUDA 12.8); TRL's vLLM-touching modules import
# 0.12+-only symbols. We never use TRL's internal vLLM path — rollouts run on
# our own external vLLM server via the rollout_func seam — so force this False
# to make GRPOTrainer importable. Do not "fix" by upgrading vLLM here.
def is_vllm_available(*_args, **_kwargs):  # noqa: F811
    return False
PATCH
    log "patched ${IMPORT_UTILS}"
fi

# 4. Commit + push the branch to the fork.
git -c user.email="${GIT_AUTHOR_EMAIL:-trl-fork@megagem.local}" \
    -c user.name="${GIT_AUTHOR_NAME:-megagem-trl-fork}" \
    commit --quiet -am "MegaGem: force is_vllm_available()->False for vLLM 0.10.2 (the RL plan §5 #6)"
FORK_SHA="$(git rev-parse HEAD)"
log "patched commit: ${FORK_SHA}"

FORK_URL="https://github.com/${GH_USER}/trl.git"
git remote add fork "${FORK_URL}"
log "pushing ${BRANCH} -> ${GH_USER}/trl"
git push --quiet --force fork "${BRANCH}"

# 5. Pin pyproject.toml to the exact fork SHA (reproducible — not a moving
#    branch ref). Matches the single `trl @ git+...trl.git@...` line.
NEW_REF="trl @ git+https://github.com/${GH_USER}/trl.git@${FORK_SHA}"
python3 - "$PYPROJECT" "$NEW_REF" <<'PY'
import re, sys
path, new_ref = sys.argv[1], sys.argv[2]
src = open(path).read()
patched, n = re.subn(
    r'"trl @ git\+https://github\.com/[^"]+"',
    '"' + new_ref + '"',
    src,
)
if n != 1:
    sys.exit(f"expected exactly 1 trl git ref in pyproject.toml, found {n}")
open(path, "w").write(patched)
print(f"pyproject.toml pinned -> {new_ref}")
PY

log "DONE."
log "Fork:    ${FORK_URL%.git}/tree/${BRANCH}"
log "Pinned:  ${FORK_SHA}"
log "Next:    review pyproject.toml diff, commit it, then 'uv sync --extra rl'"
