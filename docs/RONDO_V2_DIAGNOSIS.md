# Rondo v2 diagnosis

Measured before changing the football engine, 3 November 2026. Ten initial scenarios, seeds 42–51, with the original default physics. Full per-episode measurements are in `v2-baseline-diagnostics.json` and `v2-legacy-policy-diagnostics.json`.

| Diagnostic | Scripted possession | Existing 4096-step policy |
|---|---:|---:|
| Mean completed passes | 16.2 | 3.5 |
| Interceptions per episode | 0 | 1.0 |
| Mean possession duration | 21.09s | 4.16s |
| Mean pass distance | 6.59m | 6.79m |
| Mean pass travel time | 0.649s | 0.528s |
| Mean nearest defender-to-ball distance | 5.32m | 3.91m |
| Adjacent perimeter passes | 100% | 100% |
| Mean normalized proximity pressure | 0 | 0.047 |
| Mean open geometric passing lanes | 4.63 / 5 | 4.20 / 5 |

Both used defender maximum speed 2.3m/s, acceleration 6m/s², reaction delay 0.15s and ball speed 10m/s. Metrics also include eight pass-direction bins, receiver/pair counts, defender movement, simultaneous chasing and an optimistic from-rest time to reach a pass segment. That reach estimate is a diagnostic, not a claim that a defender chose that target.

## Causes in the implementation

1. Six anchors sit on a large hexagon, with a narrow radial support allowance. The default away-from-pressure/tangent movement pushes players outward and adds automatic tangential movement. This favors circulation around a ring.
2. The scripted receiver score pays twice the current lane clearance, adds receiver clearance, and subtracts pass distance. An unopposed adjacent player is consistently the best choice. The learned smoke policy also concentrated on adjacent players, but lost possession; weak training did not cause the scripted baseline's absence of pressure.
3. The nearest defender pursues the current ball with only 0.1s velocity prediction. A 6.6m pass arrives in roughly 0.65s, while defenders remain over 5m from the ball. Chasing cannot close that gap. Cover targets are chosen using distance to the *other* defender, which can bunch both defenders around the same side.
4. Body orientation only delays a release until a hard-coded 0.45s holding threshold, then permits release regardless of angle. Reception preparation does not persist as physical state. This makes repeat circulation cheap.
5. Swept ball-versus-static-position tests prevent basic tunneling, but ignore player motion over the physics tick and order hits by closest approach rather than first circle entry. Improve the collision geometry without guaranteeing outcomes.
6. Moderate reward gives +1 per pass but only +0.02 per second. A safe pass every 1.2s dominates possession-time reward. There is no reward for tactical diversity, which is appropriate; the physical problem must become harder instead.
7. Existing PPO does control the advertised actions and saves real weights. Its alternating mode freezes copied opponents, but only keeps five recent in-memory neural snapshots, cycles deterministically through them, and lacks scripted pool entries, persistent generations, optimizer/RNG/environment resume, or checkpoint-pair evaluation.
8. The SVG viewer interpolates actual 20 Hz state recordings. It does not invent the circular strategy or the ball travel. The weakness is in the football task and baselines, not animation.

## Implementation response

Preserve v1 mechanics and old weights in explicit legacy playback. Introduce an environment-v2 12m rectangular drill with 15m easy/10m hard presets, perimeter regions, controllable support, receiving/preparation/orientation state and relative swept collisions. Calibrate at the **same 2.3m/s defender speed first**. Use observable ball motion to seek feasible interceptions and dynamic press/cover assignments. Rebalance time/pass/turnover rewards without directional bonuses. Add persistent immutable opponent populations, alternating generations, matched evaluation, compatible resume and honest UI controls.

Legacy checkpoint inputs/actions keep their v1 interpretation. New checkpoints carry environment/observation/action versions; v1 weights cannot silently enter v2 training. Stored v1 replays remain playable without re-simulation or model migration.

## Calibrated v2 result

After the redesign, the same ten seeds with **unchanged defender speed 2.3m/s** produced 9.2 mean completed passes, 0.6 interceptions per episode, 17.43s mean duration, 3.89m mean nearest-defender distance and 34.8% adjacent passes. Mean proximity pressure increased from 0 to 0.061. Full measured outputs: `v2-grid-diagnostics.json`.

The selected default uses 7m/s ground passes, 0.32s receiving preparation and 0.12s pass preparation. This is a tunable toy-football calibration. A controlled trial of the new 12m geometry with the old 10m/s ball produced 0.3 interceptions per episode; changing to 7m/s plus receiving preparation produced 0.6. The effect is not monotonic: an 8m/s trial changed receiver choices and produced no interceptions across the ten seeds. We did not guarantee outcomes or select interception events for animation.

Pass-distance diagnostics measure the intended target's distance at release. Travel times terminate at the next observed reception/failure/interception/out-of-bounds event, which can include an unintended teammate receiving the pass. They should not be interpreted as one-to-one distance/speed pairs. Available lanes in the offline diagnostic use geometric clearance; the v2 learning/evaluation lane-margin feature additionally estimates physically reachable defender motion.
