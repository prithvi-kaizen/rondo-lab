"""Matched held-out scenarios and checkpoint-versus-checkpoint evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from rondolab.policies.neural import load_checkpoint
from rondolab.replay import generate_episode
from rondolab.simulation.config import Config


def matchup(possession=None, defending=None, config=None, seeds=None, survival_seconds=15., passes_target=10,
            possession_metadata=None, defending_metadata=None, baseline='heuristic'):
    seeds = list(range(10001, 10006)) if seeds is None else list(seeds)
    if not seeds:
        raise ValueError('Evaluation requires at least one seed')
    config = config or Config()
    pm = possession_metadata or {'team': 'possession', 'scenario': 'adaptive', 'environment_version': 2}
    dm = defending_metadata or {'team': 'pressing', 'scenario': 'adaptive', 'environment_version': 2}
    rows = []
    for seed in seeds:
        if possession is not None:
            row = generate_episode(config, seed, possession, pm, baseline, defending, False, dm)
        elif defending is not None:
            row = generate_episode(config, seed, defending, dm, baseline=baseline, record=False)
        else:
            row = generate_episode(config, seed, baseline=baseline, record=False)
        rows.append(row)
    passes = np.array([r['passes'] for r in rows])
    durations = np.array([r['duration'] for r in rows])
    successes = [r['duration'] >= survival_seconds or r['passes'] >= passes_target for r in rows]
    turned = [r['duration'] for r in rows if r['turnovers']]
    regained = [r['time_to_regain'] for r in rows if r.get('time_to_regain') is not None]
    result = {'episodes': len(rows), 'seeds': seeds, 'mean_passes': float(passes.mean()),
              'median_passes': float(np.median(passes)), 'longest_sequence': int(passes.max()),
              'passes_std': float(passes.std(ddof=1)) if len(rows) > 1 else 0.,
              'passes_ci95_normal': float(1.96 * passes.std(ddof=1) / len(rows) ** .5) if len(rows) > 1 else 0.,
              'mean_duration': float(durations.mean()),
              'pass_completion': sum(r['passes'] for r in rows) / max(1, sum(r['attempts'] for r in rows)),
              'interceptions': sum(r['interceptions'] for r in rows),
              'interception_frequency': float(np.mean([r['interceptions'] for r in rows])),
              'turnovers': sum(r['turnovers'] for r in rows),
              'forced_turnover_rate': float(np.mean([r.get('forced_turnovers', r['interceptions']) for r in rows])),
              'mean_time_to_regain': float(np.mean(regained)) if regained else None,
              'mean_time_to_turnover': float(np.mean(turned)) if turned else None,
              'regain_censored_episodes': len(rows) - len(regained),
              'tackles': sum(r.get('tackles', 0) for r in rows),
              'success_rate': float(np.mean(successes)),
              'success_definition': f'Possession survives {survival_seconds:g}s OR completes {passes_target} passes.',
              'episodes_detail': rows,
              'flags': sorted({flag for row in rows for flag in row.get('flags', [])}),
              'note': 'Normal-approximation CI across scenario seeds, not independent training seeds. '
                      'Regain means direct defensive contact. Other terminal losses are separate turnovers.'}
    for key in ['average_pressure', 'average_carrier_pressure', 'pressure_fraction', 'average_nearest_defender_distance',
                'average_open_lanes', 'coordination_fraction', 'both_chasing_fraction',
                'adjacent_pass_fraction', 'average_pass_distance', 'average_ball_travel_time',
                'successful_split_passes', 'pressured_receptions', 'central_interceptions', 'distinct_players']:
        result[key] = float(np.mean([r.get(key, 0) for r in rows]))
    result['pass_direction_bins'] = np.sum([r.get('pass_direction_bins', [0] * 8) for r in rows], axis=0).tolist()
    result['receiver_distribution'] = np.sum([r['receivers'] for r in rows], axis=0).tolist()
    return result


def matrix(possession_paths, defending_paths, config=None, seeds=None):
    def load(path, team):
        if path in (None, 'scripted'):
            return None, None
        model, meta = load_checkpoint(path)
        if meta['team'] != team:
            raise ValueError(f'Expected a {team} checkpoint')
        if meta['environment_version'] != 2:
            raise ValueError('Matchup matrix uses v2 policies; old policies remain in legacy playback')
        return model, meta
    ps = [(p, *load(p, 'possession')) for p in possession_paths]
    ds = [(d, *load(d, 'pressing')) for d in defending_paths]
    rows = []
    for p, pm, pmeta in ps:
        for d, dm, dmeta in ds:
            rows.append({'possession': str(p or 'scripted'), 'defending': str(d or 'scripted'),
                         'possession_generation': (pmeta or {}).get('generation'),
                         'defending_generation': (dmeta or {}).get('generation'),
                         **matchup(pm, dm, config, seeds, possession_metadata=pmeta, defending_metadata=dmeta)})
    return {'config': (config or Config()).model_dump(), 'seeds': seeds or list(range(10001, 10006)),
            'matchups': rows, 'environment_version': 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--possession', nargs='+', default=['scripted'])
    p.add_argument('--defending', nargs='+', default=['scripted'])
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--config')
    p.add_argument('--output', default='experiments/matchups.json')
    args = p.parse_args()
    config = Config.model_validate_json(Path(args.config).read_text()) if args.config else Config()
    result = matrix(args.possession, args.defending, config, list(range(10001, 10001 + args.episodes)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps([{k: r[k] for k in ['possession', 'defending', 'mean_passes', 'mean_duration', 'success_rate']}
                      for r in result['matchups']], indent=2))


if __name__ == '__main__':
    main()
