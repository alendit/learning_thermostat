"""Tests for solar exposure feature construction."""

from __future__ import annotations

from custom_components.learning_thermostat.solar import (
    SOLAR_PROFILE_BIN_AZIMUTHS,
    build_solar_profile,
)


def test_south_facing_bin_is_strongest_for_southern_sun() -> None:
    """Direct exposure should peak at the matching compass bin."""
    profile = build_solar_profile(
        solar_flux=500.0,
        sun_azimuth=180.0,
        sun_elevation=45.0,
    )

    assert profile.flux == 500.0
    assert profile.features[0] == 500.0
    strongest_bin = max(
        range(len(SOLAR_PROFILE_BIN_AZIMUTHS)),
        key=lambda index: profile.directional[index],
    )
    assert SOLAR_PROFILE_BIN_AZIMUTHS[strongest_bin] == 180.0
    assert profile.directional[strongest_bin] > 0.0


def test_directional_bins_are_zero_below_horizon() -> None:
    """Night-time sun positions should not create direct window gain."""
    profile = build_solar_profile(
        solar_flux=120.0,
        sun_azimuth=180.0,
        sun_elevation=-5.0,
    )

    assert profile.flux == 120.0
    assert profile.features[0] == 120.0
    assert all(value == 0.0 for value in profile.directional)


def test_missing_sun_position_falls_back_to_diffuse_only() -> None:
    """Solar flux remains usable when the sun entity or forecast position is missing."""
    profile = build_solar_profile(
        solar_flux=80.0,
        sun_azimuth=None,
        sun_elevation=None,
    )

    assert profile.flux == 80.0
    assert profile.features[0] == 80.0
    assert all(value == 0.0 for value in profile.directional)
