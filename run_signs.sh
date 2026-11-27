#!/usr/bin/env bash
set -e
cd "$HOME/scene_understanding"
signs_vlm=false
signs_dry=false
for signs_arg in "$@"; do
 [[ "$signs_arg" != '--vlm' ]] || signs_vlm=true
 [[ "$signs_arg" != '--dry-run' ]] || signs_dry=true
done
if [[ "$signs_vlm" == true && "$signs_dry" == false && -z "${ANTHROPIC_API_KEY:-}" ]]; then
 read -r -s -p 'Anthropic API key (hidden, not saved): ' ANTHROPIC_API_KEY
 printf '\n'
 export ANTHROPIC_API_KEY
fi
exec .venv/bin/python sign_vlm.py "$@"
