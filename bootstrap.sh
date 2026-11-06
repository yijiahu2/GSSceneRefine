#!/usr/bin/env bash
# ONE GSSceneRefine environment: create once, then maintain dependencies IN PLACE.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENV_NAME=gsrefine
MODE=install
case "${1:-}" in
  "") ;;
  --check) MODE=check ;;
  --sync-deps) MODE=sync ;;
  --update-env) MODE=update ;;
  -h|--help)
    cat <<'HELP'
Usage: bash bootstrap.sh [--check | --sync-deps | --update-env]
  (default)      Create gsrefine if missing; synchronize required Node/UI packages.
  --check        Verify Python/Node/UI versions and run tests without installing.
  --sync-deps    Install/update ONLY required Node 22 and UI npm packages in gsrefine.
  --update-env   Update all packages using environment.yml, without pruning.
No option recreates gsrefine or creates a second Conda environment.
HELP
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda unavailable in this shell; initialize Conda first." >&2; exit 1
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[env] Reusing $ENV_NAME"
  if [[ "$MODE" == update ]]; then
    conda env update -n "$ENV_NAME" -f environment.yml
  fi
else
  if [[ "$MODE" == check || "$MODE" == sync ]]; then
    echo "[env] $ENV_NAME missing; create with: conda env create -f environment.yml" >&2
    exit 1
  fi
  conda env create -f environment.yml
fi

NODE_MAJOR="$(conda run -n "$ENV_NAME" node -p 'parseInt(process.versions.node.split(".")[0],10)' 2>/dev/null || true)"
if [[ ! "$NODE_MAJOR" =~ ^[0-9]+$ ]] || (( NODE_MAJOR < 22 || NODE_MAJOR >= 23 )); then
  if [[ "$MODE" == check ]]; then
    echo "[deps] Node.js 22 required by SplatTransform 3.x; gsrefine has '${NODE_MAJOR:-missing}'." >&2
    exit 1
  fi
  echo "[deps] Updating Node.js in the EXISTING gsrefine environment (not recreating it)"
  conda install -n "$ENV_NAME" -c conda-forge 'nodejs>=22,<23' -y
fi

UI_DEP=ui/node_modules/playcanvas/build/playcanvas.mjs
SOG_DEP=ui/node_modules/.bin/splat-transform
if [[ ! -f "$UI_DEP" || ! -f "$SOG_DEP" ]]; then
  if [[ "$MODE" == check ]]; then
    echo "[deps] Missing PlayCanvas or SplatTransform; run bash bootstrap.sh --sync-deps" >&2
    exit 1
  fi
  echo "[deps] Installing project-local UI packages from ui/package.json"
  conda run -n "$ENV_NAME" npm --prefix ui install --no-audit --no-fund
else
  echo '[deps] PlayCanvas and SplatTransform are present'
fi

if [[ "$MODE" == check || "$MODE" == install || "$MODE" == update ]]; then
  conda run -n "$ENV_NAME" python run.py doctor --no-config
  conda run -n "$ENV_NAME" python -m pytest -q tests
  conda run -n "$ENV_NAME" node --test ui/tests/*.test.mjs
fi
printf '\nReady: cd %s && conda activate %s\n' "$PWD" "$ENV_NAME"
