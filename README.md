# Rondo Lab v2

A local football and competitive reinforcement-learning laboratory. Watch a physical 6v2 rondo, train possession and pressing policies against frozen opponents, and compare their actual decisions across generations.

The default is a **12 × 12m rectangular training grid**. Easy is 15m; hard is 10m. Outside players adjust within perimeter regions. Defenders press, cover and pursue interception points or likely receivers inferred from the **visible ball trajectory**. No hidden upcoming pass target is supplied to defenders.

## Browser demo

The static simulation gallery includes three scripted scenarios, real generation 0/5/10 policy recordings, synchronized comparisons, tactical overlays and timeline controls. It needs no Python backend or model download. Training remains available in the local lab.

```bash
npm ci --prefix frontend
npm run build:demo --prefix frontend
python3 -m http.server 4173 --directory frontend/demo-dist
```

See [public demo hosting and provenance](docs/PUBLIC_DEMO.md). The GitHub Pages workflow builds and publishes the gallery on main pushes once Pages uses GitHub Actions.

## Launch

Requires Python 3.11+, uv, Node.js 22+ and npm. Verified on Apple Silicon with Python 3.13.5 / Node 24.

```bash
uv sync
npm ci --prefix frontend
./scripts/dev.sh
```

Open [Rondo Lab](http://127.0.0.1:5173). Backend API documentation: [port 8000/docs](http://127.0.0.1:8000/docs). The scripted drill starts immediately without models. Continuous session advances to another real seeded episode after a turnover; disable it to inspect one episode. Space pauses outside form controls; right arrow steps a frame. Pitch focus removes surrounding panels.

For one local server after building:

```bash
npm run build --prefix frontend
uv run uvicorn rondolab.api:app --host 127.0.0.1 --port 8000
```

Open [the built app](http://127.0.0.1:8000). Restart the backend after the first build so the static mount is registered. This is a localhost application without network authentication.

## What changed

The old heuristic passed exclusively to adjacent players, averaged 5.32m from the nearest defender and suffered no interceptions in ten diagnostic episodes. The v2 calibration used the **same 2.3m/s defender speed** and produced interceptions in 7/10 initial and 7/10 held-out scenarios. Adjacent passing fell to about 42% in the initial scenarios. This is a tunable toy-football balance, not a proven optimal drill.

The full [diagnosis](docs/RONDO_V2_DIAGNOSIS.md) separates formation, receiver selection, ball/reception timing, defender targeting, rewards and weak training. [Verification](docs/V2_VERIFICATION.md) records tests, actual training outcomes, browser checks and limitations.

## Adversarial Training Lab

Use the dedicated sidebar page to:

- Train possession, defenders or alternating teams in a separate local process.
- Select Smoke, Standard or Extended budgets; change steps, environments, block length, reward design, curriculum, evaluation count and compute device.
- Inspect which team is learning, which is frozen, the sampled opponent and the recorded population.
- Pause or stop at a safe PPO block boundary and save a full session. Resume compatible sessions with both optimizers, RNG state and historical opponents restored.
- Select a **population run** and an actual recorded generation. Watch a possession/defending pair from the same seed, or evaluate a bounded checkpoint-versus-checkpoint matrix.
- Run geometry, defensive coordination, opponent adaptation, generation and reward-policy comparisons. Reward comparisons use actual selected policy weights; they do not fabricate ablations or retrain automatically.

Generations 0, 5 and 10 exist locally in the verified `v2-release` run. Checkpoints are generated files and stay out of Git. On a clean clone, checkpoint lists are empty until training runs. Missing stages are explicitly identified. Select the population run before selecting generations; this prevents confusing unrelated training lineages.

Watch AI Learn supports a fixed opposing checkpoint and synchronized split-screen possession comparisons. Inspection shows recorded observations/masks, chosen and previous actions, genuine probabilities/value estimates when available, pressure, receiving readiness, controller targets and interception attempts. Scripted policies explicitly have no neural probabilities.

## Training commands

```bash
# Reproduce the verified ten-generation adversarial run
uv run python -m rondolab.train --mode both --scenario adaptive \
  --timesteps 5120 --block-steps 256 --num-envs 2 --evaluation-episodes 2 \
  --output experiments/v2-release

# Short end-to-end check (4 generations with these defaults)
uv run python -m rondolab.train --mode both --scenario adaptive --preset smoke \
  --output experiments/my-smoke

# Fixed-opponent possession / pressing runs
uv run python -m rondolab.train --mode possession --scenario adaptive \
  --timesteps 512 --block-steps 128 --num-envs 2 --seed 72 \
  --output experiments/my-possession
uv run python -m rondolab.train --mode pressing --scenario adaptive \
  --timesteps 512 --block-steps 128 --num-envs 2 --seed 73 \
  --output experiments/my-pressing

# Carrier decisions with rule-based off-ball support
uv run python -m rondolab.train --mode possession --scenario basic \
  --timesteps 4096 --output experiments/basic-passing

# Moderate / extended experiments; quality is not guaranteed
uv run python -m rondolab.train --mode both --scenario adaptive --preset standard \
  --curriculum-stage 5 --output experiments/standard --device auto
uv run python -m rondolab.train --mode both --scenario adaptive --preset extended \
  --curriculum-stage 1 --auto-curriculum --evaluation-episodes 10 \
  --output experiments/extended-curriculum

# Resume to a larger TOTAL budget, preserving original optimizer/settings/population
uv run python -m rondolab.train --resume experiments/my-smoke/session.pt \
  --timesteps 4096 --output experiments/my-smoke-continued
```

Use a new output directory for every run/resume. Existing actor checkpoints are never overwritten. Initial weights are saved before optimization. Each block saves an immutable actor snapshot, `session.pt`, `population.json`, `metrics.json` and atomic `status.json`. `session.pt` is the latest resumable boundary within that run; resuming writes to a new run directory and retains the original session.

Smoke = 2048 decisions, blocks of 256, two worlds. Standard = 100,000 / 1024 / four worlds. Extended = 1,000,000 / 2048 / four worlds. All are configurable. Physics runs serially per world, with batched neural inference and separate per-environment/per-agent GAE traces. A step counts a world decision, not an agent sample.

The local worker lock prevents conflicting CLI/API training processes. Programmatic `train()` calls are intended for controlled experiments/tests. Pause/stop requests finish the current rollout/update/evaluation and checkpoint it. An abrupt crash can lose an incomplete block; the previous complete session remains available. CPU is verified; CUDA/MPS selection is implemented but was not benchmarked here.

## Evaluation and replay

```bash
# Scripted state recording and football diagnostic flags
uv run python -m rondolab.demo --seed 44 --output examples/v2-scripted.json
uv run python -m rondolab.diagnostics --episodes 10 --output experiments/diagnostics.json

# Actual checkpoint-pair replay
uv run python -m rondolab.replay \
  --checkpoint experiments/v2-release/possession-004864.pt \
  --opponent-checkpoint experiments/v2-release/pressing-005120.pt \
  --seed 42 --output examples/v2-generation10.json

# Matched G0 / G5 / G10 evaluation (nine pairings, same held-out seeds)
uv run python -m rondolab.matchups \
  --possession experiments/v2-release/possession-000000.pt \
               experiments/v2-release/possession-002304.pt \
               experiments/v2-release/possession-004864.pt \
  --defending experiments/v2-release/pressing-000000.pt \
              experiments/v2-release/pressing-002560.pt \
              experiments/v2-release/pressing-005120.pt \
  --episodes 5 --output experiments/generation-matrix.json

# Reproduce fixed-team smoke baselines and the stored evaluation report
# Requires the 5120-step v2-release run above.
uv run python scripts/verify_v2.py --run experiments/v2-release
```

`scripted` can replace a path in either matrix list. Optional `--config configs/standard_v2.json` fixes evaluation physics explicitly. Per-update **fixed benchmark** evaluation uses the original run's drill configuration and cover/press defenders, independent of whichever opponent/geometry was sampled for training. Historical-opponent and curriculum evaluations are separate. Evaluation cannot update weights.

Possession success means surviving 15 seconds **or** completing ten passes. Duration, completion, interceptions, pressure and passing distributions remain visible so success rate cannot hide poor football. Interception counts/frequencies, censored regain times, receiver choices, completed passing networks, pressured receptions, successful split passes, defender movement and coordination proxies come from state/event logs. See metric definitions in [architecture](docs/ARCHITECTURE.md).

**Observed learning quality:** the short runs verify PPO, freezing, sampling, checkpointing, resume and evaluation. They do **not** demonstrate useful improvement by both teams. Later possession policies can stall until the holding limit. Learned defenders can beat earlier neural possession while failing against the strong scripted baseline. These failures are visible in the matrix and diagnostic flags. Full measured results are in [V2_EVALUATION.json](docs/V2_EVALUATION.json). Longer runs and multiple independent training seeds are required before claiming tactical learning.

## Physics, actions and rewards

Fixed 20 Hz physics and default 5 Hz decisions. Ball passes travel continuously with resistance and first relative-circle-entry collision tests against moving players. Receiving delay depends on orientation/pressure; release requires preparation and a finite body turn. Poor control can leave a loose ball. Defenders use finite acceleration/turning and predict feasible contacts from current ball motion. An unsuccessful interception attempt does not guarantee a turnover.

V2 possession actions: 0 hold, 1–6 pass to an eligible absolute receiver ID, 7/8 adjust left/right in the assigned region, 9 offer a receiving angle, 10 separate from pressure, 11 return to anchor. Basic passing trains the carrier only, with scripted support. Adaptive learning trains all six. Defenders: press, cover, block one of six receiver lanes, recover, pursue observable ball motion, hold shape or close a receiver. Both teams have shared weights within their own team, separate local critics and no GAN loss.

| Reward | Sparse | Moderate | Heavy |
|---|---:|---:|---:|
| Possession time per second | 0 | +0.10 | +0.10 |
| Completed legal pass | 0 | +0.08 | +0.40 |
| Turnover | -4 default | -4 default | -4 default |
| Pressure shaping per decision | 0 | 0 | bounded negative, max magnitude 0.02 |

Defending receives the opposite valid team reward. Invalid actions cost only the submitting actor 0.05; masks prevent such choices during PPO sampling. There is no direction, diversity or split-pass bonus. Holding limits and finite horizons prevent unlimited retention rewards. One/two-touch constraints approximate release windows of 0.85/1.5 seconds in v2; they are explicit drill rules, not modeled foot contacts.

The six curriculum stages cover basic passing, pressure, adaptive support, defending, adversarial populations and generalization. Manual selection is supported. Automatic promotion needs at least ten held-out episodes and team-specific football thresholds; stage 5 requires qualification by both teams before stage 6. Time elapsed is never sufficient. Stage 6 varies size/opponent strategy during training while retaining a fixed benchmark.

## Compatibility

- Environment v1 and its 90-feature policy interpretation are retained in `rondolab/legacy`.
- V2 actors use 110 features and versioned action semantics. Old v1 checkpoints play in explicit legacy mode; they cannot resume v2 training.
- Stored replay schema v1 remains readable; schema v2 adds controller/region/readiness/interception information and opposing-policy metadata. Stored states need no model to play.
- Unpublished development-draft v2 snapshots remain on disk. Those without finalized engine provenance are excluded from the normal policy/session catalog; `/api/policies?include_drafts=true` exposes their metadata. This avoids presenting drafts as final-generation evidence.

## Checks and limitations

```bash
uv run pytest -q
uv run ruff check .
npm run build --prefix frontend
```

Tests cover physics, API contracts, Gymnasium/PettingZoo validation, hidden-intention exclusion, masks, team rewards, immutable opponents, alternating updates, same-device deterministic resume, generation metadata, matched scenarios, legacy playback and cooperative API pause/resume. CI runs Python checks and the TypeScript production build.

This is a steering-based toy simulation. Team physical profiles are shared; there are no aerial balls, stamina, biomechanics or learned continuous pass strength. Players stay within assigned regions; full position exchanges are not implemented. Lane/coordination/pressure metrics are defined proxies, not tactical truth. Collision separation is approximate. No quantitative browser FPS or physical-phone touch benchmark is claimed. GPU behavior and cross-device bitwise resume are unverified.

The repo keeps simulation, environment contracts, PPO, population/evaluation, API workers and rendering separate. Rendering interpolates recorded states and never determines physics or learning outcomes. Architecture and future-agent requirements are in `docs/ARCHITECTURE.md` and `AGENTS.md`.
