# Rondo Lab

Keep simulation, environment contracts, RL, API and rendering separate. Use SI units and seeded deterministic simulation. Never invent learning stages, curves, evaluations or policy explanations. Add meaningful tests for physics changes and run Gymnasium/PettingZoo validators for environment changes. Keep experiment seeds and checkpoint provenance in outputs. Preserve versioned replay compatibility and document changes. Record actual limitations rather than presenting placeholders as features. Keep docs and CLI examples executable. Make small reviewable commits at coherent milestones when requested.

Design direction is in docs/ARCHITECTURE.md and the original product brief. Apply the installed antislop core, UI, human, copywriting and mobile skills during implementation (user selected this mode). Do not change instruction files for skill installation without explicit authorization.

## V2 invariants

Preserve explicit v1 playback and schema-v1 import. Do not reinterpret v1 weights as 110-feature v2 actors. Keep environment, action and engine provenance in checkpoints; retain immutable historical opponents and versioned full sessions. Training must alternate against frozen models and sample latest/history/scripted opponents with nonzero historical support. Never backpropagate through opponent simulator actions or introduce a GAN loss.

Keep fixed benchmark conditions independent of sampled training opponents, curriculum and generalization geometry; log historical/curriculum evaluation separately. Matchup matrices must use the same held-out seeds/configuration. Detect stalling, co-chasing, reward/benchmark disagreement and pair/direction concentration without banning legitimate strategies. Do not claim bilateral improvement from smoke rewards.

Test same-runtime session resume against uninterrupted training, fingerprints of the frozen team, moving-relative swept collision and hidden-intention exclusion. Basic passing must retain scripted off-ball support and force no-op policy gradients during flight. Update docs/RONDO_V2_DIAGNOSIS.md and docs/V2_VERIFICATION.md when football balance changes. Keep generated weights/replays outside Git. Leave an explicitly user-requested running local app available after QA.
