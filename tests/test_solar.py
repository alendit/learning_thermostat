"""Tests for solar exposure feature construction."""

from __future__ import annotations

from custom_components.learning_thermostat.solar import (
    compass_label,
    build_solar_orientation_features,
    build_solar_sample,
)


def test_facade_exposure_is_strongest_when_sun_matches_orientation() -> None:
    """Direct exposure should peak at the matching learned facade."""
    sample = build_solar_sample(
        solar_flux=500.0,
        sun_azimuth=270.0,
        sun_elevation=45.0,
    )

    west_features = build_solar_orientation_features(sample, 270.0)
    south_features = build_solar_orientation_features(sample, 180.0)

    assert west_features[0] == 500.0
    assert west_features[1] > south_features[1]
    assert west_features[1] > 0.0


def test_perpendicular_facade_exposure_supports_corner_rooms() -> None:
    """The second facade should represent the orientation plus 90 degrees."""
    sample = build_solar_sample(
        solar_flux=300.0,
        sun_azimuth=180.0,
        sun_elevation=30.0,
    )

    features = build_solar_orientation_features(sample, 90.0)

    assert features[0] == 300.0
    assert features[1] == 0.0
    assert features[2] > 0.0


def test_directional_exposure_is_zero_below_horizon() -> None:
    """Night-time sun positions should not create direct window gain."""
    sample = build_solar_sample(
        solar_flux=120.0,
        sun_azimuth=180.0,
        sun_elevation=-5.0,
    )
    features = build_solar_orientation_features(sample, 180.0)

    assert features == (120.0, 0.0, 0.0)


def test_missing_sun_position_falls_back_to_diffuse_only() -> None:
    """Solar flux remains usable when the sun entity or forecast position is missing."""
    sample = build_solar_sample(
        solar_flux=80.0,
        sun_azimuth=None,
        sun_elevation=None,
    )
    features = build_solar_orientation_features(sample, 180.0)

    assert sample.flux == 80.0
    assert sample.azimuth is None
    assert sample.elevation is None
    assert features == (80.0, 0.0, 0.0)


def test_compass_labels_render_near_cardinal_azimuths() -> None:
    """Debug output should use human-readable compass directions."""
    assert compass_label(270.0) == "west"
    assert compass_label(268.0) == "west"
    assert compass_label(224.0) == "southwest"
