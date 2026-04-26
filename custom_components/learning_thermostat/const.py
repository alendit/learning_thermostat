"""Constants for the learning thermostat integration."""

from __future__ import annotations

from homeassistant.const import Platform

DOMAIN = "learning_thermostat"
NAME = "Learning Thermostat"

PLATFORMS = [Platform.CLIMATE, Platform.SENSOR]

CONF_UNDERLYING_ENTITY_IDS = "underlying_entity_ids"
CONF_TEMPERATURE_SENSOR_ENTITY_ID = "temperature_sensor_entity_id"
CONF_OUTDOOR_ENTITY_ID = "outdoor_entity_id"
CONF_OUTDOOR_TEMPERATURE_SENSOR_ENTITY_ID = "outdoor_temperature_sensor_entity_id"
CONF_FORECAST_ENTITY_ID = "forecast_entity_id"
CONF_CYCLE_MINUTES = "cycle_minutes"
CONF_TARGET_TEMPERATURE = "target_temperature"
CONF_MIN_TEMPERATURE = "min_temperature"
CONF_MAX_TEMPERATURE = "max_temperature"
CONF_LEARNING_LOOKAHEAD_STEPS = "learning_lookahead_steps"
CONF_LEARNING_HISTORY_STEPS = "learning_history_steps"
CONF_LEARNING_L2_REG = "learning_l2_reg"
CONF_LEARNING_ERROR_DECAY = "learning_error_decay"
CONF_LEARNING_RATE = "learning_rate"
CONF_LEARNING_ENERGY = "learning_energy"
CONF_LEARNING_MODEL = "learning_model"

LEARNING_MODEL_LINEAR = "linear"
LEARNING_MODEL_PHYSICAL = "physical"
LEARNING_MODEL_LABELS = {
    LEARNING_MODEL_LINEAR: "Linear",
    LEARNING_MODEL_PHYSICAL: "Physical",
}
DEFAULT_LEARNING_MODEL = LEARNING_MODEL_LINEAR

SERVICE_RESET_LEARNING = "reset_learning"
SERVICE_TRIGGER_TICK = "trigger_tick"

DEFAULT_NAME = "Underfloor Heating"
DEFAULT_CYCLE_MINUTES = 15
DEFAULT_TARGET_TEMPERATURE = 21.0
DEFAULT_MIN_TEMPERATURE = 15.0
DEFAULT_MAX_TEMPERATURE = 28.0

DATA_STORAGE = "storage"
DATA_SERVICES_REGISTERED = "services_registered"
DATA_RUNTIMES = "runtimes"
