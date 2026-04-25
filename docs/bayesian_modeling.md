# Bayesian Modeling Notes

These are cliff notes for a later design discussion about how the thermostat
should relate physical reality, model structure, learned parameters, and the
data we can actually observe.

## Core Framing

The thermostat does not observe the true thermal system. It observes sparse,
noisy traces: room temperature, outside temperature, heating state, target
temperature, and weather-derived solar flux. The model is therefore not the
house. It is a compact explanation of the data that is useful for control.

There are three distinct layers that should not be conflated:

- Reality: heat capacity, underfloor lag, solar gain through windows, internal
  gains, ventilation, walls, clouds, sensors, and occupants.
- Model structure and hyperparameters: which variables exist, how many lags are
  used, whether solar direction is represented by compass bins, a continuous
  facade direction, or something else.
- Learned parameters: the actual fitted weights or gains inside the selected
  structure.

Bayesian thinking is useful because most of the difficult questions are not
"what is the best coefficient?" They are questions about evidence, priors,
identifiability, uncertainty, and model complexity.

## Solar Direction Example

Solar flux alone is incomplete because the same surface irradiance can have
different room impact depending on sun azimuth, elevation, window orientation,
shading, and the room envelope.

The current directional-bin approach uses continuous sun position to project
solar flux onto fixed vertical-window basis directions. It does not add new
physical measurements. It adds model structure: a prior that the room's solar
response can be approximated as a weighted combination of several facade
directions.

That structure is flexible, but it adds many parameters:

- one diffuse solar channel per lag
- eight directional solar channels per lag
- therefore eight additional learned weights per lag compared with scalar solar

This is a bias-variance tradeoff. The model can represent multiple facades, but
it needs more data and can learn noise when the thermostat history is sparse.

## Alternative Solar Structures

Possible alternatives form a complexity ladder.

1. Scalar solar

   Use only raw solar flux. This is compact and stable, but cannot distinguish
   morning sun from evening sun or infer window orientation.

2. Two signed directional components

   Use continuous east-west and north-south components derived from sun position.
   This adds two parameters per lag and can learn one dominant direction as a
   vector. It is compact, but opposite exposures can cancel and multiple facades
   are hard to represent.

3. Fixed directional basis

   Use several compass-oriented half-wave exposure features. This keeps the
   optimizer linear and can represent multiple exposures, but increases
   dimensionality.

4. Learned orientation

   Learn a structural parameter such as facade angle `theta`, then project solar
   flux onto `theta` and perhaps `theta + 90deg`. This is closer to the physical
   story for a room with one dominant facade or corner exposure, but the
   orientation is a nonlinear parameter. A practical implementation would search
   candidate orientations, fit the linear model for each, and compare evidence or
   validation error.

5. Hierarchical or mixture model

   Treat room exposure type as latent: diffuse-only, single facade, corner room,
   through-room, or multi-facade. This is more Bayesian in spirit but likely too
   heavy for the current integration unless simpler models fail.

## Priors And Identifiability

The model should encode what is plausible before seeing data:

- Heating should usually increase future room temperature, with lag.
- Outside temperature usually pulls room temperature toward the envelope loss
  direction.
- Solar gain should usually be non-negative, but observed correlations can look
  negative because of thermostat behavior, night/day schedules, ventilation, or
  confounding.
- A single thermostat zone is usually dominated by one facade or two adjacent
  facades, not many independent directions.
- More parameters require more history and stronger regularization.

Some parameters may not be identifiable from available data. For example, if the
heating is almost always off during sunny periods, or the target temperature is
flat, the data may not distinguish passive solar gain from ordinary outdoor
temperature effects.

## Evidence And Model Selection

A useful Bayesian question is not only "which fitted model has the lowest error?"
It is "does the additional structure earn its complexity?"

Candidate comparison criteria:

- out-of-sample prediction error over recent history
- improvement during solar-active periods specifically
- posterior or bootstrap stability of the learned direction
- sign and magnitude plausibility of learned effects
- whether control decisions change in a useful way
- whether the model remains stable after recorder gaps or short histories

The integration probably does not need full Bayesian inference initially. But it
can still use Bayesian ideas:

- prefer simpler structures unless data supports complexity
- expose confidence separately from point estimates
- avoid treating learned orientation as fact
- keep diagnostics clear about uncertainty and model evidence

## Diagnostics Implications

Diagnostics should distinguish:

- current feature contribution: what the observed current inputs contribute to
  the prediction now
- learned sensitivity: what would happen if a controllable variable changed
- structural inference: what facade direction or exposure type the model thinks
  is plausible
- confidence/evidence: how stable and useful that inference appears to be

For controlled variables such as heating state, current contribution can be zero
when the heat is currently off. That does not mean heating has no learned effect.
The better diagnostic is a what-if comparison such as predicted next temperature
with heat off versus heat on.

For solar direction, displaying too many raw feature names can make the model
look more physically certain than it is. A better long-term presentation may
show an aggregate solar effect, a learned dominant direction if justified, and a
confidence or evidence score.

## Open Questions

- What is the simplest solar structure that produces better control decisions
  than scalar solar?
- Should orientation be a learned nonlinear structural parameter instead of a
  set of fixed basis weights?
- Should solar gains be constrained non-negative, or is regularized signed
  linear fitting good enough given confounding?
- How should model confidence be represented in Home Assistant attributes?
- Should the optimizer compare several model structures and select the simplest
  one with enough evidence?
- How much history is required before enabling directional solar diagnostics?
- Which diagnostics help debugging without implying false precision?

## Working Hypothesis

The selected implementation direction is to treat solar exposure as compact
model-structure learning rather than adding more fixed linear compass-bin
features:

- keep scalar diffuse solar as the baseline
- use one compact two-facade directional model
- search a dominant facade orientation on a 5 degree grid when enough
  solar-active data exists
- represent the perpendicular facade as `orientation + 90deg`
- keep the internal orientation training buffer out of Home Assistant
  attributes
- expose the orientation as diagnostic, with confidence, not as user-facing
  configuration

The diagnostic surface should render sorted positive directional contribution
shares, for example `west 90%, south 10%`, rather than exposing the internal
candidate-grid mechanics as if they were physical facts.
