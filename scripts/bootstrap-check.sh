#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .drone.yml ]]; then
  echo "missing .drone.yml"
  exit 1
fi

if [[ ! -d custom_components ]]; then
  echo "No custom_components directory yet; bootstrap repo is valid."
  exit 0
fi

component_count=0

for dir in custom_components/*; do
  [[ -d "$dir" ]] || continue
  component_count=$((component_count + 1))
  name="$(basename "$dir")"

  manifest="$dir/manifest.json"
  init_file="$dir/__init__.py"
  if [[ ! -f "$manifest" ]]; then
    echo "missing manifest: $manifest"
    exit 1
  fi

  if [[ ! -f "$init_file" ]]; then
    echo "missing module entrypoint: $init_file"
    exit 1
  fi

  manifest_domain="$(jq -r '.domain' "$manifest")"
  manifest_name="$(jq -r '.name' "$manifest")"
  manifest_version="$(jq -r '.version' "$manifest")"
  config_flow_enabled="$(jq -r '.config_flow // false' "$manifest")"

  if [[ -z "$manifest_domain" || "$manifest_domain" == "null" ]]; then
    echo "manifest domain is empty: $manifest"
    exit 1
  fi

  if [[ "$manifest_domain" != "$name" ]]; then
    echo "manifest domain $manifest_domain does not match directory $name"
    exit 1
  fi

  if [[ -z "$manifest_name" || "$manifest_name" == "null" ]]; then
    echo "manifest name is empty: $manifest"
    exit 1
  fi

  if [[ -z "$manifest_version" || "$manifest_version" == "null" ]]; then
    echo "manifest version is empty: $manifest"
    exit 1
  fi

  if [[ "$config_flow_enabled" == "true" && ! -f "$dir/config_flow.py" ]]; then
    echo "config_flow enabled but config_flow.py is missing: $dir/config_flow.py"
    exit 1
  fi
done

echo "Validated $component_count custom component(s)."
