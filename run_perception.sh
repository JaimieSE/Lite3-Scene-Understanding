#!/usr/bin/env bash
set -e
cd "$HOME/scene_understanding"
perception_vlm=false
perception_dry=false
for perception_arg in "$@"; do
 [[ "$perception_arg" != '--vlm' ]] || perception_vlm=true
 [[ "$perception_arg" != '--dry-run' ]] || perception_dry=true
done
if [[ "$perception_vlm" == true && "$perception_dry" == false && -z "${ANTHROPIC_API_KEY:-}" ]]; then
 read -r -s -p 'Anthropic API key (hidden, not saved): ' ANTHROPIC_API_KEY
 printf '\n'
 export ANTHROPIC_API_KEY
fi
exec .venv/bin/python combined_perception.py "$@"
