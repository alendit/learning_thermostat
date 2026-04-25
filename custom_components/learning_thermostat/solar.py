"""Solar exposure feature construction for learning models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

SOLAR_ORIENTATION_FEATURE_SIZE = 3
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
COMPASS_LABELS: tuple[str, ...] = (
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
)


@dataclass(frozen=True, slots=True)
class SolarSample:
    """Raw irradiance and sun-position input for one sample."""

    flux: float
    azimuth: float | None
    elevation: float | None

    def as_dict(self) -> dict[str, float | None]:
        """Return a stable JSON-serializable representation."""
        return {
            "flux": self.flux,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
        }


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


def build_solar_sample(
    *,
    solar_flux: Any,
    sun_azimuth: Any,
    sun_elevation: Any,
) -> SolarSample:
    """Build one raw solar sample from HA irradiance and sun-position values."""
    return SolarSample(
        flux=_coerce_non_negative_float(solar_flux),
        azimuth=_coerce_float(sun_azimuth),
        elevation=_coerce_float(sun_elevation),
    )


def diffuse_only_sample(solar_flux: Any) -> SolarSample:
    """Return a solar sample with flux but no usable sun position."""
    return build_solar_sample(
        solar_flux=solar_flux, sun_azimuth=None, sun_elevation=None
    )


def coerce_solar_sample(value: Any, fallback_flux: Any = None) -> SolarSample:
    """Coerce caller input into a raw solar sample."""
    if isinstance(value, SolarSample):
        return value
    if isinstance(value, dict):
        return build_solar_sample(
            solar_flux=value.get("flux", fallback_flux),
            sun_azimuth=value.get("azimuth"),
            sun_elevation=value.get("elevation"),
        )
    if isinstance(value, (list, tuple)) and value:
        # Legacy solar-profile rows do not contain sun position. Preserve only
        # the diffuse/raw irradiance channel.
        return diffuse_only_sample(value[0])
    return diffuse_only_sample(fallback_flux)


def build_solar_orientation_features(
    sample: SolarSample | dict[str, Any] | None,
    orientation_azimuth: Any,
) -> tuple[float, float, float]:
    """Build diffuse plus two perpendicular facade-exposure features."""
    coerced = coerce_solar_sample(sample)
    orientation = _coerce_float(orientation_azimuth)
    if orientation is None:
        orientation = 0.0
    facade_1 = _facade_exposure(coerced, orientation)
    facade_2 = _facade_exposure(coerced, orientation + 90.0)
    return (coerced.flux, facade_1, facade_2)


def compass_label(azimuth: Any) -> str | None:
    """Return the nearest eight-way compass label for an azimuth."""
    value = _coerce_float(azimuth)
    if value is None:
        return None
    index = int(round((_normalize_azimuth(value) % 360.0) / 45.0)) % len(COMPASS_LABELS)
    return COMPASS_LABELS[index]


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


def _facade_exposure(sample: SolarSample, orientation_azimuth: float) -> float:
    """Return projected direct exposure for one vertical facade orientation."""
    if (
        sample.azimuth is None
        or sample.elevation is None
        or sample.elevation <= 0.0
        or sample.flux <= 0.0
    ):
        return 0.0
    elevation_factor = math.cos(math.radians(sample.elevation))
    azimuth_factor = math.cos(
        math.radians(_angle_delta(sample.azimuth, orientation_azimuth))
    )
    exposure = sample.flux * max(0.0, elevation_factor * azimuth_factor)
    return 0.0 if abs(exposure) < 1e-9 else exposure


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


def _normalize_azimuth(value: float) -> float:
    """Normalize an azimuth into Home Assistant's north-clockwise degrees."""
    return value % 360.0
