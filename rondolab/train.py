import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from rondolab.environments.parallel import RondoParallelEnv, action_mask, observation
from rondolab.matchups import matchup
from rondolab.policies.heuristics import scripted_actions
from rondolab.policies.neural import ActorCritic, save_checkpoint
from rondolab.rl.curriculum import level_config, promotion
from rondolab.rl.population import OpponentPool, fingerprint
from rondolab.rl.ppo import gae, update
from rondolab.simulation.config import Config

BUDGETS = {'smoke': {'timesteps': 2048, 'block_steps': 256, 'num_envs': 2},
           'standard': {'timesteps': 100000, 'block_steps': 1024, 'num_envs': 4},
           'extended': {'timesteps': 1000000, 'block_steps': 2048, 'num_envs': 4}}


def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def train(mode=None, timesteps=4096, seed=42, output='experiments/v2-run', scenario=None,
          checkpoint_interval=2048, config=None, learning_rate=3e-4, gamma=.99, clip=.2,
          entropy=.01, device='cpu', block_steps=256, num_envs=1, resume=None,
          curriculum_stage=None, auto_curriculum=False, evaluation_episodes=2,
          pool_weights=(.4, .35, .25)):
    if timesteps < 1 or min(checkpoint_interval, block_steps, num_envs, evaluation_episodes) < 1:
        raise ValueError('Training budgets must be positive')
    torch.set_num_threads(1)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('*.pt')):
        raise ValueError('Output already contains checkpoints. Choose a new output directory to preserve history.')
    restored = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if restored and (restored.get('kind') != 'training-session' or restored.get('environment_version') != 2):
        raise ValueError('Resume requires a compatible v2 training session.pt, not legacy or actor-only weights')
    if restored and restored['settings']['config'].get('engine_revision') != '2.0':
        raise ValueError('This draft session predates the finalized v2 football mechanics; start a new v2 run')
    if restored:
        saved = restored['settings']
        mode, scenario, seed = saved['mode'], saved['scenario'], saved['seed']
        c = Config(**saved['config'])
        learning_rate, gamma, clip, entropy = [saved[k] for k in ('learning_rate', 'gamma', 'clip', 'entropy')]
        block_steps, num_envs = saved['block_steps'], saved['num_envs']
        auto_curriculum, evaluation_episodes = saved['auto_curriculum'], saved['evaluation_episodes']
        stage = restored['curriculum_stage']
        checkpoint_interval = saved['checkpoint_interval']
        curriculum_stage = stage if saved['use_curriculum'] else None
        pool_weights = saved['pool_weights']
    else:
        mode = mode or 'possession'
        scenario = scenario or ('basic' if mode == 'possession' else 'adaptive')
        c = config or Config()
        stage = curriculum_stage or (5 if mode == 'both' else c.curriculum)
    if mode not in ('possession', 'pressing', 'both') or scenario not in ('basic', 'adaptive'):
        raise ValueError('Unsupported mode or scenario')
    if c.environment_version != 2:
        raise ValueError('New training requires environment v2. Legacy checkpoints are playback-only.')
    if mode == 'both' and scenario == 'basic':
        raise ValueError('Adversarial training requires adaptive six-player actions')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    models = {team: ActorCritic().to(device) for team in ('possession', 'pressing')}
    optimizers = {team: torch.optim.Adam(m.parameters(), lr=learning_rate) for team, m in models.items()}
    steps = updates = episodes = 0
    team_updates = {'possession': 0, 'pressing': 0}
    history, encounters = [], []
    pool = OpponentPool(weights=pool_weights)
    if restored:
        for team, model in models.items():
            model.load_state_dict(restored['models'][team])
            optimizers[team].load_state_dict(restored['optimizers'][team])
            # Optimizer tensors follow their parameter device, including MPS/CUDA resume.
            for state in optimizers[team].state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        steps, updates, episodes = restored['steps'], restored['updates'], restored['episodes']
        team_updates, history, encounters = restored['team_updates'], restored['history'], restored['encounters']
        pool = OpponentPool(restored['pool'], weights=pool_weights)
        rng.bit_generator.state = restored['numpy_rng']
        torch.set_rng_state(restored['torch_rng'])
        if device == 'cuda' and restored.get('accelerator_rng') is not None:
            torch.cuda.set_rng_state_all(restored['accelerator_rng'])
        elif device == 'mps' and restored.get('accelerator_rng') is not None:
            torch.mps.set_rng_state(restored['accelerator_rng'])
    if timesteps <= steps:
        raise ValueError(f'Resume target must exceed the saved {steps} steps')
    settings = {'mode': mode, 'scenario': scenario, 'seed': seed, 'config': c.model_dump(),
                'lineage': restored['settings'].get('lineage', Path(resume).parent.name) if restored else out.name,
                'benchmark_protocol': 'fixed-cover-v2',
                'learning_rate': learning_rate, 'gamma': gamma, 'clip': clip, 'entropy': entropy,
                'block_steps': block_steps, 'num_envs': num_envs, 'auto_curriculum': auto_curriculum,
                'checkpoint_interval': checkpoint_interval, 'use_curriculum': curriculum_stage is not None,
                'evaluation_episodes': evaluation_episodes, 'pool_weights': list(pool_weights)}
    common = {**settings, 'algorithm': 'shared-policy PPO', 'environment_version': 2,
              'observation_size': 110, 'action_version': 2, 'compatibility': 'rondo-v2',
              'engine_revision': c.engine_revision,
              'created_at': time.time(),
              'device': device, 'schema_version': 2}
    if not restored:
        for team, model in models.items():
            path = out / f'{team}-000000.pt'
            save_checkpoint(path, model, common | {'team': team, 'steps': 0, 'stage': 'untrained',
                                                   'generation': 0, 'team_generation': 0,
                                                   'opponents_encountered': [], 'evaluation': None}, optimizers[team])
            pool.add(team, path, 0, 0)
    stop_requested = [None]
    previous_handler = None
    if __import__('threading').current_thread() is __import__('threading').main_thread():
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, 'cancelled'))
    started = time.time()
    status = {}

    def save_session(state):
        payload = {'kind': 'training-session', 'environment_version': 2, 'settings': settings,
                   'models': {t: {k: v.detach().cpu() for k, v in m.state_dict().items()} for t, m in models.items()},
                   'optimizers': {t: o.state_dict() for t, o in optimizers.items()}, 'steps': steps,
                   'updates': updates, 'episodes': episodes, 'team_updates': team_updates,
                   'pool': pool.entries, 'numpy_rng': rng.bit_generator.state,
                   'torch_rng': torch.get_rng_state(), 'curriculum_stage': stage,
                   'history': history, 'encounters': encounters,
                   'accelerator_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else
                   torch.mps.get_rng_state() if device == 'mps' else None}
        temporary = out / 'session.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / 'session.pt')
        atomic_json(out / 'population.json', {'entries': pool.entries, 'sampling_weights': pool.weights})
        atomic_json(out / 'metrics.json', {'metadata': common, 'history': history, 'encounters': encounters})
        atomic_json(out / 'status.json', {**status, 'state': state, 'resume': str(out / 'session.pt')})

    try:
        while steps < timesteps:
            team = ('possession' if updates % 2 == 0 else 'pressing') if mode == 'both' else mode
            if auto_curriculum and mode == 'both' and stage < 5:
                team = 'pressing' if stage == 4 else 'possession'
            other_team = 'pressing' if team == 'possession' else 'possession'
            # Single-team runs use a fixed scripted benchmark. Alternating runs sample immutable populations.
            entry = pool.sample(other_team, rng) if mode == 'both' else next(
                e.copy() for e in pool.entries if e['id'] == ('scripted-cover' if team == 'possession' else 'scripted-possession'))
            opponent = pool.model(entry, device)
            before = {t: fingerprint(m) for t, m in models.items()}
            block_config = level_config(stage, c, rng) if curriculum_stage is not None or auto_curriculum else c
            block_scenario = 'basic' if auto_curriculum and stage <= 2 else scenario
            if team == 'possession' and entry['kind'] == 'scripted':
                block_config = block_config.model_copy(update={'defender_strategy': entry['strategy']})
            remaining = min(block_steps, checkpoint_interval, timesteps - steps)
            env_count = min(num_envs, remaining)
            # Each block starts fresh scenarios, so no trajectory or recurrent credit crosses opponents.
            envs = [RondoParallelEnv(block_config, block_scenario) for _ in range(env_count)]
            for env in envs:
                env.reset(seed=int(rng.integers(1000000, 2000000000)))
            n = 2 if team == 'pressing' else 1 if block_scenario == 'basic' else 6
            records = {k: [] for k in ('obs', 'mask', 'action', 'logp', 'value', 'reward', 'next', 'term', 'done')}
            outcomes = []
            component_totals = {}
            status = common | {'steps': steps, 'target': timesteps, 'updates': updates, 'episodes': episodes,
                               'generation': max(team_updates.values()), 'team_generations': team_updates,
                               'learning_team': team, 'frozen_team': other_team, 'sampled_opponent': entry,
                               'opponent_pool': pool.entries, 'curriculum_stage': stage,
                               'latest': history[-1] if history else None, 'state': 'running'}
            atomic_json(out / 'status.json', status)
            for _ in range(max(1, remaining // env_count)):
                def indices(world, team=team, block_scenario=block_scenario):
                    if team == 'pressing':
                        return [6, 7]
                    return [world.ball.carrier if world.ball.carrier is not None and world.ball.carrier < 6 else 0] \
                        if block_scenario == 'basic' else list(range(6))
                active = [indices(env.world) for env in envs]
                obs = np.array([[observation(env.world, i) for i in ids] for env, ids in zip(envs, active, strict=True)])
                masks = np.array([[action_mask(env.world, i, block_scenario) for i in ids]
                                  for env, ids in zip(envs, active, strict=True)])
                if team == 'possession' and block_scenario == 'basic':
                    for j, env in enumerate(envs):
                        if env.world.ball.carrier is None or env.world.ball.carrier >= 6:
                            masks[j] = 0
                            masks[j, :, 0] = 1
                with torch.no_grad():
                    dist, value = models[team](torch.tensor(obs.reshape(-1, 110), device=device),
                                               torch.tensor(masks.reshape(-1, 12), device=device))
                    action = dist.sample()
                    logp = dist.log_prob(action)
                actions = action.cpu().numpy().reshape(env_count, n)
                rewards, terminals, dones, next_values = [], [], [], []
                for j, (env, ids) in enumerate(zip(envs, active, strict=True)):
                    w = env.world
                    acts = scripted_actions(w, random=entry.get('strategy') == 'random')
                    if opponent is not None:
                        other_ids = [6, 7] if team == 'possession' else list(range(6))
                        acts.update(opponent.decisions(w, other_ids, block_scenario, deterministic=False))
                    if not (team == 'possession' and block_scenario == 'basic' and w.ball.carrier is None):
                        acts.update({i: int(a) for i, a in zip(ids, actions[j], strict=True)})
                    _, reward, term, trunc, info = env.step({f'p{i}': a for i, a in acts.items()})
                    terminal, done = term['p0'], term['p0'] or trunc['p0']
                    with torch.no_grad():
                        next_ids = indices(w)
                        _, nv = models[team](torch.tensor(np.stack([observation(w, i) for i in next_ids]), device=device),
                                              torch.tensor(np.stack([action_mask(w, i, block_scenario) for i in next_ids]), device=device))
                    rewards.append([reward[f'p{i}'] for i in ids])
                    terminals.extend([terminal] * n)
                    dones.extend([done] * n)
                    next_values.extend(nv.cpu().tolist())
                    for key, amount in info[f'p{ids[0]}']['reward_components'].items():
                        component_totals[key] = component_totals.get(key, 0.) + amount
                    if done:
                        episodes += 1
                        outcomes.append({'seed': w.seed, 'passes': w.passes, 'duration': w.time, 'reason': w.reason})
                        env.reset(seed=int(rng.integers(1000000, 2000000000)))
                for key, item in [('obs', obs.reshape(-1, 110)), ('mask', masks.reshape(-1, 12)),
                                  ('action', action.cpu().numpy()), ('logp', logp.cpu().numpy()),
                                  ('value', value.cpu().numpy()), ('reward', np.array(rewards).flatten()),
                                  ('next', next_values), ('term', terminals), ('done', dones)]:
                    records[key].append(item)
                steps += env_count
            r = {k: np.array(v) for k, v in records.items()}
            advantages, returns = gae(r['reward'], r['value'], r['next'], r['term'], r['done'], gamma)
            def tensor(data, dtype=torch.float32):
                return torch.tensor(data, dtype=dtype, device=device)
            batch = (tensor(r['obs'].reshape(-1, 110)), tensor(r['mask'].reshape(-1, 12), torch.bool),
                     tensor(r['action'].flatten(), torch.long), tensor(r['logp'].flatten()),
                     tensor(advantages.flatten()), tensor(returns.flatten()))
            losses = update(models[team], optimizers[team], batch, clip=clip, entropy_coef=entropy)
            updates += 1
            team_updates[team] += 1
            generation = max(team_updates.values())
            after = {t: fingerprint(m) for t, m in models.items()}
            if after[other_team] != before[other_team]:
                raise AssertionError('Frozen team changed during the other team update')
            heldout = list(range(10001, 10001 + evaluation_episodes))
            args = {'possession': models[team]} if team == 'possession' else {'defending': models[team]}
            learned_metadata = {'team': team, 'scenario': block_scenario, 'environment_version': 2}
            metadata_args = {'possession_metadata': learned_metadata} if team == 'possession' else {
                'defending_metadata': learned_metadata}
            benchmark_config = c.model_copy(update={'defender_strategy': 'cover_press'})
            fixed = matchup(config=benchmark_config, seeds=heldout, **args, **metadata_args)
            historical_args = {'possession': models[team], 'defending': opponent} if team == 'possession' else {
                'possession': opponent, 'defending': models[team]}
            historical_config = benchmark_config
            if team == 'possession' and entry['kind'] == 'scripted':
                historical_config = benchmark_config.model_copy(update={'defender_strategy': entry['strategy']})
            historical = matchup(config=historical_config, seeds=heldout, **historical_args, **metadata_args,
                                 baseline='random' if entry.get('strategy') == 'random' else 'heuristic')
            curriculum_config = level_config(stage, c) if curriculum_stage is not None or auto_curriculum else c
            curriculum_eval = fixed if curriculum_config == benchmark_config else matchup(
                config=curriculum_config, seeds=heldout, **args, **metadata_args)
            encountered = {'block': updates, 'generation': generation, 'learning': team, 'frozen': other_team,
                           'opponent': entry, 'outcomes': outcomes, 'evaluation': historical}
            encounters.append(encountered)
            row = {'steps': steps, 'updates': updates, 'generation': generation, 'team_generation': team_updates.copy(),
                   'team': team, 'frozen_team': other_team, 'opponent': entry, 'episodes': episodes,
                   'mean_step_reward': float(r['reward'].mean()), 'reward_components': component_totals,
                   'parameter_hashes_before': before, 'parameter_hashes_after': after,
                   'evaluation': fixed, 'historical_evaluation': historical,
                   'benchmark_config': benchmark_config.model_dump(), 'curriculum_evaluation': curriculum_eval,
                   'curriculum_stage': stage, 'environment_config': block_config.model_dump(), **losses}
            row['diagnostic_flags'] = list(fixed.get('flags', []))
            row['promotion_qualified'] = promotion(stage, curriculum_eval, team=team) > stage
            if team == 'pressing' and row['mean_step_reward'] > 0 and fixed['forced_turnover_rate'] < .1:
                row['diagnostic_flags'].append('Positive pool reward without fixed-benchmark turnovers; inspect opponent overfitting.')
            history.append(row)
            path = out / f'{team}-{steps:06d}.pt'
            save_checkpoint(path, models[team], common | {'team': team, 'steps': steps,
                            'generation': generation, 'team_generation': team_updates[team],
                            'stage': 'final' if steps >= timesteps else 'intermediate', 'updates': updates,
                            'opponents_encountered': [e['opponent']['id'] for e in encounters if e['learning'] == team],
                            'evaluation': fixed, 'config': block_config.model_dump(),
                            'note': 'Real optimizer checkpoint; generation is provenance, not playing strength.'}, optimizers[team])
            pool.add(team, path, generation, steps, fixed)
            if auto_curriculum:
                if stage == 5 and mode == 'both':
                    eligible = {r['team'] for r in history if r['curriculum_stage'] == 5 and r.get('promotion_qualified')}
                    stage = 6 if eligible == {'possession', 'pressing'} else 5
                else:
                    stage = promotion(stage, curriculum_eval, team=team)
            status = common | {'steps': steps, 'target': timesteps, 'updates': updates, 'episodes': episodes,
                               'generation': generation, 'team_generations': team_updates, 'learning_team': team,
                               'frozen_team': other_team, 'sampled_opponent': entry,
                               'opponent_pool': pool.entries, 'latest': row, 'curriculum_stage': stage,
                               'elapsed': time.time() - started, 'state': 'running'}
            request_file = out / 'control.json'
            control = json.loads(request_file.read_text()).get('action') if request_file.exists() else stop_requested[0]
            state = control if control in ('paused', 'cancelled') else 'completed' if steps >= timesteps else 'running'
            save_session(state)
            print(json.dumps({k: row[k] for k in ['steps', 'generation', 'team', 'mean_step_reward', 'entropy']}), flush=True)
            if state in ('paused', 'cancelled'):
                break
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['possession', 'pressing', 'both'])
    p.add_argument('--scenario', choices=['basic', 'adaptive'])
    p.add_argument('--preset', choices=list(BUDGETS))
    p.add_argument('--timesteps', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='experiments/v2-run')
    p.add_argument('--checkpoint-interval', type=int, default=2048)
    p.add_argument('--device', choices=['cpu', 'auto', 'cuda', 'mps'], default='cpu')
    p.add_argument('--config')
    p.add_argument('--block-steps', type=int)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--resume')
    p.add_argument('--curriculum-stage', type=int, choices=range(1, 7))
    p.add_argument('--auto-curriculum', action='store_true')
    p.add_argument('--evaluation-episodes', type=int, default=2)
    args = vars(p.parse_args())
    budget = BUDGETS[args.pop('preset') or 'smoke']
    for key in ('timesteps', 'block_steps', 'num_envs'):
        if args[key] is None:
            args[key] = budget[key]
    if args['config']:
        args['config'] = Config.model_validate_json(Path(args['config']).read_text())
    import fcntl
    lock_path = Path(os.environ.get('RONDO_WORKER_LOCK', str(Path(__file__).resolve().parent.parent /
                                                          'experiments/.training-worker.lock')))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another local training worker is running. Pause or stop it before starting another.')
        train(**args)


if __name__ == '__main__':
    main()
