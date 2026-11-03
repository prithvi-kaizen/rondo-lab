# Rondo Lab implementation plan

1. Deterministic NumPy world at 20 Hz: acceleration, bounded turning, constrained support motion, collisions, rolling ball and swept interception. Record complete episodes from scripted policies.
2. Eight-agent PettingZoo ParallelEnv plus a Gym passing wrapper. Run upstream compatibility validators and deterministic physics tests.
3. Shared actor/value networks per team. Collect masked on-policy rollouts, GAE, clipped PPO, entropy and value losses. Save actual initialization and update checkpoints. Evaluate on disjoint seeds.
4. Add support actions and pressing actions using the same physical controller. Alternate updates against an immutable opponent pool.
5. React/TypeScript SVG pitch interpolates recorded 20 Hz states using requestAnimationFrame. Playback never changes physics. FastAPI generates episodes and manages one separate training subprocess.
6. Run all tests, CPU smoke training for every mode, evaluation and browser QA of scripted and saved-policy playback. Record evidence and limitations.

## Boundaries

`simulation` owns all football mechanics and interpretable units. `environments` owns observations, masks and rewards. `policies` owns scripted heuristics and networks. `rl` owns optimizer state, rollouts and curriculum. `replay` owns versioned stored states. API passes validated configuration to those modules. The frontend only interpolates and inspects recorded states.

## Learning formulation

Each team has one shared decentralized actor/value network. Every agent sees its own relative observation; the critic is local, not centralized. Samples from simultaneous agents remain separate trajectories for advantage computation. Possession actions 0 hold, 1..6 pass to absolute player ID (self masked), 7 clockwise support, 8 counterclockwise, 9 open angle, 10 away from pressure, 11 recover anchor. Defenders use 0 press, 1 cover, 2..7 block receiver lane, 8 recover, 9 pursue ball. Same shape for both teams; unused defender actions are masked. Basic passing trains only the carrier. Adaptive possession trains all six.

Competition alternates team updates against a frozen opponent sampled from saved policy states. No opponent receives optimizer updates during another team's rollout. Evaluation uses seeds 10001 onward; training starts at the requested seed. Initial checkpoints are random neural policies, not scripted baselines.

## Visual direction from the brief

Dark football analysis tool, pitch first, restrained color, readable type and fluid animation. ENERGY 2 / RHYTHM 2 / MOTION 3. Muted green turf gives football context. White and red shirts distinguish teams through labels as well as color. A lime accent marks playback and selected controls. Pitch markings and dashed pass geometry form the visual motif. Panel borders separate controls; shadows only lift players off turf. All animation serves recorded motion; there are no decorative entrance animations.
