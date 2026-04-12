#!/usr/bin/env bash
set -euo pipefail

tag="${DRONE_TAG:-}"
version="${tag#v}"

mkdir -p dist

if [[ ! -d custom_components ]]; then
  echo "No custom_components directory yet; nothing to package."
  exit 0
fi

shopt -s nullglob

packaged=0
catalog_file="$(mktemp)"
printf '[]' > "$catalog_file"

for dir in custom_components/*; do
  [[ -d "$dir" ]] || continue

  name="$(basename "$dir")"
  manifest="$dir/manifest.json"

  if [[ ! -f "$manifest" ]]; then
    echo "missing manifest: $manifest"
    exit 1
  fi

  if [[ -n "$version" ]]; then
    manifest_version="$(jq -r '.version' "$manifest")"
    if [[ "$manifest_version" != "$version" ]]; then
      echo "tag version $version does not match $name manifest version $manifest_version"
      exit 1
    fi
  fi

  (
    cd custom_components
    zip -r "../dist/${name}-${tag:-dev}.zip" "$name" >/dev/null
  )

  checksum="$(sha256sum "dist/${name}-${tag:-dev}.zip" | awk '{print $1}')"
  manifest_version="$(jq -r '.version' "$manifest")"

  updated_catalog="$(mktemp)"
  jq \
    --arg name "$name" \
    --arg version "$manifest_version" \
    --arg file "${name}-${tag:-dev}.zip" \
    --arg sha256 "$checksum" \
    '. + [{domain: $name, version: $version, artifact: $file, sha256: $sha256}]' \
    "$catalog_file" > "$updated_catalog"
  mv "$updated_catalog" "$catalog_file"

  packaged=$((packaged + 1))
done

jq '{components: .}' "$catalog_file" > dist/catalog.json
rm -f "$catalog_file"

echo "Packaged $packaged custom component(s)."
