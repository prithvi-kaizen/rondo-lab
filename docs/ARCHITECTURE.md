# Rondo Lab v2 architecture

## Boundaries and versions

`simulation` owns physical/tactical controllers in SI units. `environments` owns the PettingZoo/Gymnasium contract, masks and rewards. `policies` contains scripted football and shared actor/local-critic networks. `rl` owns PPO, curriculum and opponent populations. `matchups` and `evaluate` run held-out football without updates. `replay` stores versioned full states. FastAPI owns local subprocess lifecycle and safe experiment paths. React/SVG interpolates those states; rendering never drives physics or fabricates learning.

`World` returns the frozen v1 engine for `environment_version=1`. V1 observations remain 90 features, legacy clockwise/counterclockwise actions keep their old interpretation, and schema-v1 state replays remain readable. V2 uses 110 features and rectangular support actions. Finalized v2 engine provenance is recorded in configuration/metadata. Old actor weights are not silently adapted to different observations or resumed into v2 optimizers. Development-draft snapshots lacking final provenance stay on disk but outside the default catalog.

## Physical state

Default grid 12m square; easy 15m, hard 10m. Four corner anchors and two opposing-side anchors have bounded axis-aligned regions. Movement actions adjust within a region; idle off-ball actions actually hold. Scripted support selects observable receiving angles. Position exchange is intentionally not implemented.

Acceleration, speed and body turning are bounded. Passing requires receiving preparation, control close to feet, pass preparation and orientation. Release leads only currently visible receiver velocity within its region. First-touch difficulty uses incoming speed/orientation/pressure. Poor touches slow the ball and temporarily prevent immediate self-recovery. Passes never jump to a player.

Collision uses the earliest quadratic entry of a moving ball into a moving player's relative interception/control circle. This avoids fast-ball and moving-player tunneling. Player separation is a small positional approximation. Event times identify their physics update; recorded frames are the completed update states, so collision-event seeking may precede the next fully recorded state by one 0.05s frame.

Defenders can press, cover, block any selected receiver lane, recover, pursue visible ball paths, hold shape or close a receiver. The interception estimator respects current motion, reaction time and reachable distance. If a contact is not feasible, the flight ray predicts a likely receiver from observable positions/velocities; it never reads the intended target or pending pass. Scripted roles exchange with reachability. Learned actors select their own intentions rather than inheriting mandatory press/cover roles.

## Observations/actions

V2 keeps the original 90 relative position/velocity/orientation/possession/boundary features, then adds readiness, pressure, team/context, lane-margin and support-region/partner features. A carrier can observe its own pending preparation, but defenders cannot read hidden passing intentions. There is no future-state, future action or privileged critic input. These are decentralized partially observed actors with local value heads, not centralized-critic MAPPO.

Possession: hold 0; pass to ID+1 at 1–6; adjust left/right 7/8; receiving angle 9; separation 10; anchor 11. Defending: press 0; cover 1; block receiver 2–7; recover 8; pursue observed ball 9; shape 10; receiver close-down 11. Pass strength is configurable physics, not an additional learned hybrid action.

Basic training has one virtual carrier decision maker; off-ball support stays scripted. During flight its policy action is forced to hold and has no action gradient. Adaptive possession has six independently sampled shared-policy actors; defending has two. Masks describe real legal actions and are applied both during collection and PPO loss calculation. The raw parallel environment still permits scripted support movements in basic mode.

## PPO and population training

Each team has its own 96 × 96 tanh actor/local-value network. PPO uses GAE (.99 discount, .95 lambda), clipping .2, entropy .01, four optimizer epochs, minibatches 128, value coefficient .5 and gradient norm .5. World time limits bootstrap value but stop traces at reset; turnovers terminate. Per-world/per-agent trajectories remain separate before flattening into optimization minibatches.

At each block, one team trains against one frozen population sample. World simulation is serial; neural inference is batched across agents/environments. Block boundaries reset scenarios when changing opponents. SHA-256 parameter fingerprints assert that the non-learning team did not change.

Pool composition: scripted chase/cover defenders, scripted/random possession, initialization, recent and historical neural snapshots. Sampling weights are .40 latest / .35 historical / .25 scripted. Historical sampling has uniform support plus extra probability for evaluated best snapshots, so early/weak opponents remain reachable. Snapshot files are immutable; model inference uses no gradients. Generation counts completed updates by each team; a pair generation is their maximum, with separate P/D counters retained.

A full session stores both networks/optimizers, CPU and applicable accelerator RNG, NumPy sampler state, counters, curriculum, encounter history and absolute references to retained immutable opponents. Resume starts a new output directory and recreates the next block boundary exactly on the same tested CPU runtime. It does not promise cross-device bitwise equivalence. Checkpoint loading preserves the training RNG. Cooperative pause/stop completes a block and persists a session. The POSIX worker lock prevents conflicting CLI/API workers.

## Evaluation and promotion

Training seeds are sampled outside the held-out 10001+ range. Fixed benchmark evaluation holds the run's base drill constant and uses cover/press defenders or scripted possession. Historical evaluation uses the selected frozen opponent; curriculum evaluation uses the fixed current level. These are separate measurements. Difficulty variation and sampled opponents cannot silently alter the fixed-benchmark curve.

Curriculum stages: basic passing, pressure, adaptive support, defending, populations, generalization. Automatic promotion requires at least ten evaluation scenarios and team-specific thresholds. Stage 5 requires both teams to qualify before generalization. Stage 6 samples 10/12/15m grids and defending strategies during training while retaining the fixed benchmark. No elapsed-time promotion.

Matrix evaluation uses identical configurations and held-out seeds for selected checkpoint pairs. Success = 15s survived OR ten completed passes. Mean/median/longest passes, completion, duration, interceptions and distributions remain visible. Regain time counts direct defensive contacts (in-flight interceptions or tackles); episodes without those contacts are censored. Holding-limit and boundary losses remain separate turnovers. Intervals are normal approximations across scenario seeds, not confidence about multiple independent training runs.

## Metric definitions and caveats

- Intended pass distance: origin to selected receiver at release. Travel time ends at the observed reception/failure/interception/out-of-bounds event; an unintended teammate can receive, so intended distance and observed travel time need not form distance/speed pairs.
- Receiver selection: intended targets. Receiver distribution/network: completed receptions; directed network edges connect passer to actual receiver.
- Adjacent passes: neighboring IDs on the perimeter ordering, not a judgment that a lateral pass is bad.
- Split pass: defenders lie on opposite sides of the release-to-receiver segment, both project inside it, and both are within 2.5m of it. Only retained receptions count as successful splits. No reward bonus.
- Proximity pressure: max(0, 1 - nearest defender distance / 2.5m by default). Ball-pressure averages include flight; carrier-pressure averages sample controlled possession only. Defending inspection shows applied ball pressure, not a self-distance.
- Open-lane estimate: positive clearance after subtracting interception reach and observable defender reachable motion. SVG lane lines separately show current static geometric clearance; feasible-flight overlays use the actual estimator.
- Coordination proxy: one press/pursuit plus one cover/lane/recovery action, with defenders more than 1.2m apart. Simultaneous chasing, travel distance and central interception locations are separate diagnostics; none is proof of intelligent cooperation.
- Non-degeneracy flags request inspection of direction/pair concentration, absent pressure, co-chasing, long unopposed possession, holding-limit stalling and reward/benchmark disagreement. They never prohibit valid sideways football.

## Visual delivery

Dark football-analysis framing, pitch first, restrained turf/white/red/lime palette. ENERGY 2 / RHYTHM 2 / MOTION 3. Shadows separate shirts/ball from turf; stripes identify the pitch. requestAnimationFrame interpolates position and shortest-angle orientation between stored frames. Continuity, trajectories, support regions, probability widths and interception predictions derive from recorded state. Shirt strides are visualization-only. The app shows missing data, unavailable probabilities, failed jobs and weak policies honestly.
