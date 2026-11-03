import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from rondolab.environments.parallel import RondoParallelEnv, action_mask, observation
from rondolab.policies.heuristics import scripted_actions
from rondolab.policies.neural import ActorCritic, save_checkpoint
from rondolab.rl.ppo import gae, update
from rondolab.simulation.config import Config


def train(mode='possession', timesteps=4096, seed=42, output='experiments/run', scenario='basic',
          checkpoint_interval=2048, config=None, learning_rate=3e-4, gamma=.99, clip=.2,
          entropy=.01, device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    c = config or Config()
    env = RondoParallelEnv(c, scenario)
    env.reset(seed=seed)
    models = {team: ActorCritic().to(device) for team in ('possession', 'pressing')}
    optimizers = {team: torch.optim.Adam(m.parameters(), lr=learning_rate) for team, m in models.items()}
    # Frozen snapshots are independent modules, never optimizer targets.
    pools = {team: [copy.deepcopy(model).eval()] for team, model in models.items()}
    steps, updates, episode = 0, 0, 0
    started = time.time()
    history = []
    common = {'mode': mode, 'scenario': scenario, 'seed': seed, 'config': c.model_dump(),
              'algorithm': 'shared-policy PPO', 'device': device, 'schema_version': 1}
    for team, model in models.items():
        if mode != 'both' and team != mode:
            continue
        save_checkpoint(out / f'{team}-000000.pt', model, common | {'team': team, 'steps': 0, 'stage': 'untrained'})
        model.to(device)
    next_checkpoint = checkpoint_interval
    while steps < timesteps:
        team = ('possession' if updates % 2 == 0 else 'pressing') if mode == 'both' else mode
        model, optimizer = models[team], optimizers[team]
        opponent_team = 'pressing' if team == 'possession' else 'possession'
        opponent = pools[opponent_team][updates % len(pools[opponent_team])] if mode == 'both' else None
        records = {k: [] for k in ('obs', 'mask', 'action', 'logp', 'value', 'reward', 'next', 'term', 'done')}
        indices = [6, 7] if team == 'pressing' else list(range(6))
        basic = team == 'possession' and scenario == 'basic'
        for _ in range(min(256, timesteps - steps)):
            w = env.world
            active = [w.ball.carrier if w.ball.carrier is not None and w.ball.carrier < 6 else 0] if basic else indices
            obs = np.stack([observation(w, i) for i in active])
            mask = np.stack([action_mask(w, i, scenario) for i in active])
            with torch.no_grad():
                dist, value = model(torch.tensor(obs, device=device), torch.tensor(mask, device=device))
                action = dist.sample()
                logp = dist.log_prob(action)
            acts = scripted_actions(w)
            if opponent is not None:
                other = list(range(6)) if team == 'pressing' else [6, 7]
                acts.update(opponent.decisions(w, other, scenario, deterministic=False))
            acts.update({i: int(a) for i, a in zip(active, action.cpu())})
            _, reward, term, trunc, _ = env.step({f'p{i}': a for i, a in acts.items()})
            terminal, done = term['p0'], term['p0'] or trunc['p0']
            next_active = [w.ball.carrier if w.ball.carrier is not None and w.ball.carrier < 6 else 0] if basic else active
            with torch.no_grad():
                _, next_values = model(torch.tensor(np.stack([observation(w, i) for i in next_active]), device=device),
                                       torch.tensor(np.stack([action_mask(w, i, scenario) for i in next_active]), device=device))
            for key, item in [('obs', obs), ('mask', mask), ('action', action.cpu().numpy()),
                              ('logp', logp.cpu().numpy()), ('value', value.cpu().numpy()),
                              ('reward', [reward[f'p{i}'] for i in active]), ('next', next_values.cpu().numpy()),
                              ('term', [terminal] * len(active)), ('done', [done] * len(active))]:
                records[key].append(item)
            steps += 1
            if done:
                episode += 1
                env.reset(seed=seed + episode)
        r = {k: np.array(v) for k, v in records.items()}
        advantages, returns = gae(r['reward'], r['value'], r['next'], r['term'], r['done'], gamma)
        def tensor(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)
        batch = (tensor(r['obs'].reshape(-1, r['obs'].shape[-1])),
                 tensor(r['mask'].reshape(-1, 12), torch.bool), tensor(r['action'].flatten(), torch.long),
                 tensor(r['logp'].flatten()), tensor(advantages.flatten()), tensor(returns.flatten()))
        metrics = update(model, optimizer, batch, clip=clip, entropy_coef=entropy)
        updates += 1
        row = {'steps': steps, 'updates': updates, 'team': team, 'episodes': episode,
               'mean_step_reward': float(r['reward'].mean()), **metrics}
        # Real fixed-seed evaluation at each update, without optimizer updates.
        from rondolab.evaluate import evaluate
        evaluation = evaluate(model=model, team=team, scenario=scenario, config=c, seeds=[10001, 10002])
        row['evaluation'] = evaluation
        history.append(row)
        (out / 'metrics.json').write_text(json.dumps({'metadata': common, 'history': history}, indent=2))
        status = common | {'steps': steps, 'updates': updates, 'episodes': episode,
                           'elapsed': time.time() - started, 'target': timesteps, 'latest': row}
        (out / 'status.json.tmp').write_text(json.dumps(status))
        (out / 'status.json.tmp').replace(out / 'status.json')
        print(json.dumps(row), flush=True)
        if steps >= next_checkpoint or steps == timesteps:
            for name, net in models.items():
                if mode != 'both' and name != mode:
                    continue
                save_checkpoint(out / f'{name}-{steps:06d}.pt', net, common | {
                    'team': name, 'steps': steps, 'updates': updates,
                    'stage': 'final' if steps == timesteps else 'intermediate',
                    'note': 'Training checkpoint; stage is provenance, not a claim of playing strength.'},
                    optimizers[name])
                net.to(device)
            next_checkpoint += checkpoint_interval
        if mode == 'both':
            pools[team].append(copy.deepcopy(model).eval())
            pools[team] = pools[team][-5:]
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['possession', 'pressing', 'both'], default='possession')
    p.add_argument('--scenario', choices=['basic', 'adaptive'], default='basic')
    p.add_argument('--timesteps', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='experiments/run')
    p.add_argument('--checkpoint-interval', type=int, default=2048)
    p.add_argument('--device', choices=['cpu', 'auto', 'cuda', 'mps'], default='cpu')
    p.add_argument('--config', help='JSON configuration file')
    args = vars(p.parse_args())
    if args['timesteps'] < 1 or args['checkpoint_interval'] < 1:
        p.error('timesteps and checkpoint interval must be positive')
    if args['config']:
        args['config'] = Config.model_validate_json(Path(args['config']).read_text())
    train(**args)


if __name__ == '__main__':
    main()
