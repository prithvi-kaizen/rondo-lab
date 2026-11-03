import argparse
import json

import numpy as np

from rondolab.policies.neural import load_checkpoint
from rondolab.replay import generate_episode
from rondolab.simulation.config import Config


def evaluate(checkpoint=None, model=None, team='possession', scenario='basic', config=None,
             seeds=None, baseline='heuristic'):
    seeds = seeds or list(range(10001, 10011))
    metadata = None
    if checkpoint:
        model, metadata = load_checkpoint(checkpoint)
        team, scenario = metadata['team'], metadata['scenario']
        config = config or Config(**(metadata['config'] | {'environment_version': metadata['environment_version']}))
    elif model:
        metadata = {'team': team, 'scenario': scenario}
    results = [generate_episode(config, seed, model, metadata, baseline=baseline, record=False) for seed in seeds]
    passes = np.array([r['passes'] for r in results])
    durations = np.array([r['duration'] for r in results])
    std = float(passes.std(ddof=1)) if len(passes) > 1 else 0.
    return {'episodes': len(seeds), 'seeds': seeds, 'mean_passes': float(passes.mean()),
            'median_passes': float(np.median(passes)), 'longest_sequence': int(passes.max()),
            'passes_std': std, 'passes_ci95_normal': 1.96 * std / len(passes) ** .5,
            'mean_duration': float(durations.mean()),
            'pass_completion': sum(r['passes'] for r in results) / max(1, sum(r['attempts'] for r in results)),
            'turnovers': sum(r['turnovers'] for r in results),
            'interceptions': sum(r['interceptions'] for r in results),
            'out_of_bounds': sum(r['out_of_bounds'] for r in results), 'episodes_detail': results,
            'note': 'CI is a normal approximation across scenario seeds, not independent training runs. '
                    'Pressing efficiency = interceptions per simulated second.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--baseline', choices=['heuristic', 'simple', 'random'], default='heuristic')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--output')
    args = p.parse_args()
    result = evaluate(checkpoint=args.checkpoint, baseline=args.baseline,
                      seeds=list(range(10001, 10001 + args.episodes)))
    data = json.dumps(result, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(data)
    print(data)


if __name__ == '__main__':
    main()
