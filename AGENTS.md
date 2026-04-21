# AGENTS.md

This repository contains a standalone Home Assistant custom integration for a
learning thermostat focused on hydronic underfloor heating controlled through
simple on-off switches.

## Goal

- Extract `learning_thermostat` from the current `versatile_thermostat`
  implementation into an independent integration.
- Keep only the features required for this integration to work well as a
  dedicated thermostat for on-off switching.
- Target underfloor heating as the primary use case.
- Exclude infrastructure and features from `versatile_thermostat` that are not
  required for this focused integration.
- Do not implement or carry translation support.

The target outcome is a small, understandable integration that solves the
actual thermostat problem well without inheriting unrelated framework code.

## Product Boundaries

- Supported control model:
  - on-off switches only
- Primary domain:
  - underfloor heating
- Non-goals for this repository unless explicitly reconsidered later:
  - translation infrastructure
  - broad HVAC device abstractions
  - support for every thermostat strategy from `versatile_thermostat`
  - speculative generalization for unknown future device classes

When deciding whether to keep or add code, prefer removing or avoiding anything
that does not directly support the focused thermostat behavior above.

## Engineering Principles

- Follow good software engineering practice.
- Do not overengineer.
- Do not future-proof speculative requirements.
- Use architecture to make future changes easier and the structure easier to
  understand.
- Prefer loose coupling between components.
- Prefer tight cohesion within each component.
- Dependencies should point from less stable parts to more stable parts.
- Keep the core behavior and decision logic as stable and framework-independent
  as practical.
- Keep Home Assistant specific wiring at the boundaries.
- Prefer explicit, readable code over clever abstractions.
- Prefer the simplest design that supports the current requirements and tests
  well.

## Architecture Guidance

Use this section to evaluate decomposition, dependency direction, side-effect
placement, interface design, and testability.

- Separate the domain logic from Home Assistant integration glue.
- Keep thermostat decision logic, learning behavior, and state handling in
  focused modules with clear responsibilities.
- Treat entities, service registration, setup code, and config flow as adapter
  layers around the core behavior.
- Avoid large utility modules that accumulate unrelated logic.
- Avoid hidden cross-module state.
- Pass dependencies explicitly where practical.
- Introduce abstractions only when they remove real duplication or make testing
  materially easier.

Before adding a new layer or interface, justify it with a concrete benefit in
this repository, not with a hypothetical future need.

### Hard Constraints

- Keep core policy and decision logic isolated from side effects.
- Keep dependencies flowing from unstable code toward stable code.
- Do not mix unrelated domains into one coordinating component.
- Prefer application services over putting business logic into entities,
  config flow, service handlers, or other Home Assistant adapter code.
- Do not introduce abstraction unless it makes a concrete likely change easier.
- Do not create shared interfaces that implementations can only satisfy by
  narrowing behavior, ignoring requirements, or throwing.

### Required Self-Check For Design-Sensitive Changes

For design-sensitive changes, include these review questions in the final note,
PR description, or equivalent handoff:

1. What component owns this behavior, and why?
2. Does this change increase or reduce coupling?
3. Did any dependency start pointing the wrong way?
4. Could any side effect be moved outward into an adapter or shell?
5. Are the abstractions semantically real?
6. Does any interface force an implementation to ignore, narrow, or reject
   required behavior?
7. What is the next likely change this makes easier?
8. What tests prove the core behavior independently of the full system?

## Development Workflow

- Development should be test-driven.
- Start each work package by defining or updating the test that captures the
  required behavior.
- Implement the smallest change that makes the test pass.
- Refactor only after behavior is covered by tests.
- Keep work packages small enough to review and reason about easily.
- Each individual work package should result in a commit.
- Keep commits focused and intentional.
- Do not mix unrelated cleanup into a feature or bug-fix commit unless
  explicitly requested.

Preferred sequence for normal work:

1. Clarify the behavior being added, changed, or removed.
2. Add or update focused tests.
3. Implement the minimal code change.
4. Run the relevant local tests.
5. Refine structure if needed while keeping tests green.
6. Commit the work package.

## Testing Expectations

- Favor fast, focused automated tests.
- Test behavior, not incidental implementation details.
- Cover thermostat decisions, learning behavior, and edge cases with unit
  tests.
- Add integration-style tests only where Home Assistant wiring or entity
  behavior needs verification.
- When extracting code from `versatile_thermostat`, add tests that protect the
  intended behavior before or during the extraction.
- Bug fixes should include a regression test when practical.

## Extraction Rules

- Prefer copying only the code that is required, then simplifying it.
- Do not preserve the source structure mechanically if a simpler structure is
  better here.
- Remove dependencies on `versatile_thermostat` infrastructure instead of
  recreating that infrastructure locally unless it is truly required.
- Rename modules, types, and concepts to fit this repository cleanly.
- Treat code from the source project as input for extraction, not as an
  architectural template that must be preserved.
- Eliminate unused options, compatibility layers, and feature flags during the
  extraction when they are not needed for the focused scope.

## CI/CD Direction

- CI/CD should follow the example of
  `/Users/dimitrivorona/projects/python/ha-codex-integration`.
- This project should have a repository in the local Gogs instance.
- This project should be integrated into local Drone.
- CI should validate structure, formatting, and tests.
- CD should deploy the custom integration into the Home Assistant instance.
- Keep the pipeline straightforward and close to the actual delivery path.
- Prefer a small number of understandable pipeline steps over elaborate build
  machinery.

Expected pipeline shape:

- bootstrap or repository sanity checks
- test environment build if needed
- automated test execution
- deployment to the Home Assistant custom components directory on mainline
- packaging or release steps only if they are actually needed

## Operational Guidance

- Prefer small, reviewable changes.
- Verify assumptions by reading the code and running tests.
- If a design choice increases indirection, complexity, or coupling, justify it
  clearly or choose the simpler option.
- Keep documentation and repository guidance in sync with the implementation.
- When in doubt, optimize for clarity of the codebase a few months from now.

## Working Standard

- The integration should be understandable without knowing the historical
  internals of `versatile_thermostat`.
- A new contributor should be able to identify:
  - the domain model
  - the thermostat control flow
  - where learning behavior lives
  - where Home Assistant specific wiring begins
- If the code structure makes those boundaries hard to see, the structure
  should be improved.
