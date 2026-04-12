#!/usr/bin/env bash
set -euo pipefail

ha_config_dir="/ha-config"
target_root="$ha_config_dir/custom_components"

if [[ ! -d custom_components ]]; then
  echo "No custom_components directory present; nothing to deploy."
  exit 0
fi

if [[ ! -d "$ha_config_dir" ]]; then
  echo "Home Assistant config mount missing at $ha_config_dir"
  exit 1
fi

if [[ -z "${HASS_SERVER:-}" || -z "${HASS_TOKEN:-}" ]]; then
  echo "HASS_SERVER and HASS_TOKEN must be provided"
  exit 1
fi

mkdir -p "$target_root"

deployed=0
for dir in custom_components/*; do
  [[ -d "$dir" ]] || continue

  name="$(basename "$dir")"
  destination="$target_root/$name"

  rm -rf "$destination"
  mkdir -p "$destination"
  cp -R "$dir"/. "$destination"/

  deployed=$((deployed + 1))
done

echo "Deployed $deployed custom component(s) into $target_root"

curl -fsS \
  -X POST \
  -H "Authorization: Bearer $HASS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' \
  "${HASS_SERVER%/}/api/services/homeassistant/restart" >/dev/null

echo "Requested Home Assistant restart."
