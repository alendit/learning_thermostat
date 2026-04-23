"""Solar exposure feature construction for learning models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

SOLAR_PROFILE_BIN_AZIMUTHS: tuple[float, ...] = (
    0.0,
    45.0,
    90.0,
    135.0,
    180.0,
    225.0,
    270.0,
    315.0,
)
SOLAR_PROFILE_SIZE = 1 + len(SOLAR_PROFILE_BIN_AZIMUTHS)


@dataclass(frozen=True, slots=True)
class SolarProfile:
    """Directional solar exposure features for one irradiance sample."""

    flux: float
    features: tuple[float, ...]

    @property
    def directional(self) -> tuple[float, ...]:
        """Return the compass-bin exposure values without the diffuse channel."""
        return self.features[1:]


def build_solar_profile(
    *,
    solar_flux: Any,
    sun_azimuth: Any,
    sun_elevation: Any,
) -> SolarProfile:
    """Build diffuse plus vertical-window directional solar exposure features."""
    flux = _coerce_non_negative_float(solar_flux)
    azimuth = _coerce_float(sun_azimuth)
    elevation = _coerce_float(sun_elevation)
    directional = [0.0] * len(SOLAR_PROFILE_BIN_AZIMUTHS)

    if azimuth is not None and elevation is not None and elevation > 0.0 and flux > 0.0:
        elevation_factor = math.cos(math.radians(elevation))
        for index, bin_azimuth in enumerate(SOLAR_PROFILE_BIN_AZIMUTHS):
            azimuth_factor = math.cos(math.radians(_angle_delta(azimuth, bin_azimuth)))
            directional[index] = flux * max(0.0, elevation_factor * azimuth_factor)

    return SolarProfile(flux=flux, features=tuple([flux, *directional]))


def diffuse_only_profile(solar_flux: Any) -> tuple[float, ...]:
    """Return a profile that preserves flux but has no directional exposure."""
    return build_solar_profile(
        solar_flux=solar_flux, sun_azimuth=None, sun_elevation=None
    ).features


def coerce_solar_profile(value: Any, fallback_flux: Any = None) -> tuple[float, ...]:
    """Coerce caller input into the fixed solar profile vector shape."""
    if isinstance(value, SolarProfile):
        return value.features
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_float(item) for item in value]
        if len(coerced) == SOLAR_PROFILE_SIZE and all(
            item is not None for item in coerced
        ):
            return tuple(float(item) for item in coerced if item is not None)
    return diffuse_only_profile(fallback_flux)


def solar_profile_flux(value: Any) -> float | None:
    """Return the diffuse/raw flux channel from a profile-like value."""
    profile = coerce_solar_profile(value)
    if not profile:
        return None
    return profile[0]


def _coerce_non_negative_float(value: Any) -> float:
    coerced = _coerce_float(value)
    if coerced is None or not math.isfinite(coerced):
        return 0.0
    return max(0.0, coerced)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _angle_delta(left: float, right: float) -> float:
    """Return the shortest signed angular distance between headings."""
    return (left - right + 180.0) % 360.0 - 180.0
