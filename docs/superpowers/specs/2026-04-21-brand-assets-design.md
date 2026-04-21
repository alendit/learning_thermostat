# Learning Thermostat Brand Assets

## Goal

Add local Home Assistant branding for the custom integration using the new
integration-local `brand/` folder so the integration shows a deliberate icon and
logo in the UI without depending on the external brands repository.

## Approved Direction

- Visual concept: balanced book + temperature symbol
- Chosen mark: `spine`
- Deliverables:
  - `icon.png`
  - `dark_icon.png`
  - `logo.png`
  - `dark_logo.png`
- Also generate matching `@2x` variants for sharper hDPI rendering

## Design

- The mark uses a rounded-square badge with:
  - a dark shell
  - a visible book spine on the left
  - open cream pages
  - a centered thermometer
- The light assets optimize for light backgrounds with dark text and shell.
- The dark assets keep the same structure but tune contrast and wordmark colors
  for darker integration surfaces.
- The logo is a horizontal composition of the mark and a two-line wordmark.

## Asset Structure

- Source assets live in `assets/brand/` as SVG files.
- Rendered Home Assistant assets live in
  `custom_components/learning_thermostat/brand/`.
- The SVG files are the canonical editable assets.
- The PNG files are tracked runtime outputs used directly by Home Assistant.

## Verification

- Render the SVG and PNG assets locally.
- Verify the Home Assistant filenames match the local brand contract:
  - `icon.png`
  - `dark_icon.png`
  - `logo.png`
  - `dark_logo.png`
- Verify icon dimensions are square and logo dimensions are landscape.
