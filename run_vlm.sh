#!/usr/bin/env bash
set -e
cd "$HOME/scene_understanding"
vlm_dry_run=false
for vlm_arg in "$@"; do
 if [[ "$vlm_arg" == "--dry-run" ]]; then vlm_dry_run=true; fi
done
if [[ "$vlm_dry_run" == false && -z "${ANTHROPIC_API_KEY:-}" ]]; then
 read -r -s -p 'Anthropic API key (hidden, not saved): ' ANTHROPIC_API_KEY
 printf '\n'
 if [[ -z "$ANTHROPIC_API_KEY" ]]; then echo 'No API key entered.'; exit 1; fi
 export ANTHROPIC_API_KEY
fi
exec .venv/bin/python periodic_vlm.py "$@"
