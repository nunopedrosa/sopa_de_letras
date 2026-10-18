#!/usr/bin/env bash
# Deploy the PWA to the web root of ramen.basecentral.link (Dreamhost).
#
# Usage (run on the server):
#   ./deploy.sh             # git pull (if this is a git checkout) + sync pwa/ -> site dir
#   ./deploy.sh --no-pull   # sync only, skip git pull
#   ./deploy.sh --dry-run   # show what would change, copy nothing
#
# Override locations via environment variables:
#   REPO_DIR=~/repos/sopa_de_letras SITE_DIR=~/ramen.basecentral.link ./deploy.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SITE_DIR="${SITE_DIR:-$HOME/ramen.basecentral.link}"

DO_PULL=1
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-pull)  DO_PULL=0 ;;
    --dry-run)  DRY_RUN=1 ;;
    -h|--help)  sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$REPO_DIR/pwa/index.html" ]]; then
  echo "Error: $REPO_DIR/pwa/index.html not found." >&2
  echo "Set REPO_DIR to the repo checkout, e.g.: REPO_DIR=~/repos/sopa_de_letras $0" >&2
  exit 1
fi

if [[ $DO_PULL -eq 1 && $DRY_RUN -eq 0 && -d "$REPO_DIR/.git" ]]; then
  echo ">> Updating repo..."
  if ! git -C "$REPO_DIR" pull --ff-only; then
    echo "!! git pull failed — deploying the current checkout as-is" >&2
  fi
fi

mkdir -p "$SITE_DIR"

# Mirror pwa/ into the site root.
# Excluded from --delete: files Dreamhost/the server may own in the web root
# (.well-known = Let's Encrypt HTTP validation, .htaccess, stats, favicon).
RSYNC_FLAGS=(
  -a --delete
  --exclude=.DS_Store
  --exclude=__pycache__
  --exclude=.git
  --exclude=.htaccess
  --exclude=.well-known
  --exclude=stats
  --exclude=favicon.ico
)
if [[ $DRY_RUN -eq 1 ]]; then RSYNC_FLAGS+=(-n -i); else RSYNC_FLAGS+=(-v); fi

echo ">> Syncing $REPO_DIR/pwa/ -> $SITE_DIR/"
rsync "${RSYNC_FLAGS[@]}" "$REPO_DIR/pwa/" "$SITE_DIR/"

if [[ $DRY_RUN -eq 0 ]]; then
  echo ">> Deployed version: $(cat "$SITE_DIR/version.json")"
fi
