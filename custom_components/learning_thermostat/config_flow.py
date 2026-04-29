"""Config flow for the learning thermostat integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.helpers import selector
from homeassistant.util import slugify

from .const import (
    CONF_CYCLE_MINUTES,
    CONF_FORECAST_ENTITY_ID,
    CONF_LEARNING_ENERGY,
    CONF_LEARNING_ERROR_DECAY,
    CONF_LEARNING_HISTORY_STEPS,
    CONF_LEARNING_L2_REG,
    CONF_LEARNING_LOOKAHEAD_STEPS,
    CONF_LEARNING_MODEL,
    CONF_LEARNING_RATE,
    CONF_MAX_TEMPERATURE,
    CONF_MIN_TEMPERATURE,
    CONF_OUTDOOR_ENTITY_ID,
    CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_TARGET_TEMPERATURE,
    CONF_TEMPERATURE_SENSOR_ENTITY_ID,
    CONF_UNDERLYING_ENTITY_IDS,
    DEFAULT_CYCLE_MINUTES,
    DEFAULT_LEARNING_MODEL,
    DEFAULT_MAX_TEMPERATURE,
    DEFAULT_MIN_TEMPERATURE,
    DEFAULT_NAME,
    DEFAULT_TARGET_TEMPERATURE,
    DOMAIN,
    LEARNING_MODEL_LABELS,
)
from .config_contract import normalize_config_entry_data
from .learning_defaults import DEFAULT_LEARNING_SWITCH_CONFIG


def _entry_unique_id(data: dict[str, Any]) -> str:
    """Build a stable unique ID for one entry."""
    switches = sorted(str(entity_id) for entity_id in data[CONF_UNDERLYING_ENTITY_IDS])
    return slugify("_".join(switches))


def _other_entry_uses_switches(
    hass, current_entry_id: str | None, candidate_unique_id: str
) -> bool:
    """Return whether another entry already uses this switch set."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == current_entry_id:
            continue
        if entry.unique_id == candidate_unique_id:
            return True
    return False


def _schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    """Build the shared schema for create and edit."""
    defaults = defaults or {}
    return vol.Schema(
        {
            vol.Required("name", default=defaults.get("name", DEFAULT_NAME)): str,
            vol.Required(
                CONF_UNDERLYING_ENTITY_IDS,
                default=defaults.get(CONF_UNDERLYING_ENTITY_IDS, []),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["switch", "input_boolean"],
                    multiple=True,
                )
            ),
            vol.Required(
                CONF_TEMPERATURE_SENSOR_ENTITY_ID,
                default=defaults.get(CONF_TEMPERATURE_SENSOR_ENTITY_ID, ""),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor"])
            ),
            vol.Optional(
                CONF_OUTDOOR_ENTITY_ID,
                default=defaults.get(
                    CONF_OUTDOOR_ENTITY_ID,
                    defaults.get(
                        CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID,
                        defaults.get(CONF_FORECAST_ENTITY_ID, ""),
                    ),
                ),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["weather", "sensor"])
            ),
            vol.Required(
                CONF_CYCLE_MINUTES,
                default=defaults.get(CONF_CYCLE_MINUTES, DEFAULT_CYCLE_MINUTES),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=5,
                    max=120,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_TARGET_TEMPERATURE,
                default=defaults.get(
                    CONF_TARGET_TEMPERATURE, DEFAULT_TARGET_TEMPERATURE
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=DEFAULT_MIN_TEMPERATURE,
                    max=DEFAULT_MAX_TEMPERATURE,
                    step=0.5,
                    unit_of_measurement="°C",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_MIN_TEMPERATURE,
                default=defaults.get(CONF_MIN_TEMPERATURE, DEFAULT_MIN_TEMPERATURE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=5,
                    max=25,
                    step=0.5,
                    unit_of_measurement="°C",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_MAX_TEMPERATURE,
                default=defaults.get(CONF_MAX_TEMPERATURE, DEFAULT_MAX_TEMPERATURE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=15,
                    max=40,
                    step=0.5,
                    unit_of_measurement="°C",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_LEARNING_LOOKAHEAD_STEPS,
                default=defaults.get(
                    CONF_LEARNING_LOOKAHEAD_STEPS,
                    DEFAULT_LEARNING_SWITCH_CONFIG["lookahead_steps"],
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=48,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_LEARNING_HISTORY_STEPS,
                default=defaults.get(
                    CONF_LEARNING_HISTORY_STEPS,
                    DEFAULT_LEARNING_SWITCH_CONFIG["history_steps"],
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=3,
                    max=48,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_LEARNING_MODEL,
                default=defaults.get(CONF_LEARNING_MODEL, DEFAULT_LEARNING_MODEL),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(value=value, label=label)
                        for value, label in LEARNING_MODEL_LABELS.items()
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_LEARNING_L2_REG,
                default=defaults.get(
                    CONF_LEARNING_L2_REG, DEFAULT_LEARNING_SWITCH_CONFIG["l2_reg"]
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=1,
                    step=0.001,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_LEARNING_ERROR_DECAY,
                default=defaults.get(
                    CONF_LEARNING_ERROR_DECAY,
                    DEFAULT_LEARNING_SWITCH_CONFIG["error_decay"],
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=1,
                    step=0.01,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_LEARNING_RATE,
                default=defaults.get(
                    CONF_LEARNING_RATE,
                    DEFAULT_LEARNING_SWITCH_CONFIG["learning_rate"],
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=1,
                    step=0.001,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_LEARNING_ENERGY,
                default=defaults.get(
                    CONF_LEARNING_ENERGY, DEFAULT_LEARNING_SWITCH_CONFIG["w_energy"]
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=1,
                    step=0.001,
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
        }
    )


class LearningThermostatConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle setup of one learning thermostat."""

    VERSION = 1

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> "LearningThermostatOptionsFlow":
        """Return the options flow."""
        return LearningThermostatOptionsFlow()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            normalized = normalize_config_entry_data(user_input)
            unique_id = _entry_unique_id(normalized)

            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            if _other_entry_uses_switches(self.hass, None, unique_id):
                errors["base"] = "already_configured"
            else:
                return self.async_create_entry(
                    title=normalized["name"],
                    data=normalized,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_schema(user_input),
            errors=errors or None,
        )


class LearningThermostatOptionsFlow(OptionsFlowWithReload):
    """Edit an existing learning thermostat."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle options editing."""
        errors: dict[str, str] = {}
        if user_input is not None:
            normalized = normalize_config_entry_data(user_input)
            unique_id = _entry_unique_id(normalized)

            if _other_entry_uses_switches(
                self.hass, self.config_entry.entry_id, unique_id
            ):
                errors["base"] = "already_configured"
            else:
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    title=normalized["name"],
                    data=normalized,
                    unique_id=unique_id,
                )
                return self.async_create_entry(title="", data={})

        defaults = user_input or dict(self.config_entry.data)
        return self.async_show_form(
            step_id="init",
            data_schema=_schema(defaults),
            errors=errors or None,
        )
