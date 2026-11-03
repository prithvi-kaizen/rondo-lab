"""Reproducible fixed-team smoke runs and matched evaluation of actual saved weights."""
import argparse
import json
from pathlib import Path

from rondolab.matchups import matchup, matrix
from rondolab.policies.neural import load_checkpoint
from rondolab.simulation.config import Config
from rondolab.train import train

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='experiments/v2-release')
    args = parser.parse_args()
    run = root / args.run
    seeds = list(range(10001, 10006))
    for mode, seed in [('possession', 72), ('pressing', 73)]:
        out = root / f'experiments/v2-final-fixed-{mode}'
        if not (out / 'session.pt').exists():
            train(mode=mode, scenario='adaptive', timesteps=512, block_steps=128, checkpoint_interval=128,
                  num_envs=2, seed=seed, output=out)
    p, pmeta = load_checkpoint(root / 'experiments/v2-final-fixed-possession/possession-000512.pt')
    d, dmeta = load_checkpoint(root / 'experiments/v2-final-fixed-pressing/pressing-000512.pt')
    results = {'seeds': seeds, 'environment': Config().model_dump(),
               'random_possession': matchup(config=Config(), seeds=seeds, baseline='random'),
               'scripted_cover_press': matchup(config=Config(), seeds=seeds),
               'scripted_chase': matchup(config=Config(defender_strategy='chase'), seeds=seeds),
               'fixed_trained_possession': matchup(p, config=Config(), seeds=seeds, possession_metadata=pmeta),
               'fixed_trained_defending': matchup(defending=d, config=Config(), seeds=seeds, defending_metadata=dmeta)}
    ps = [run / 'possession-000000.pt', run / 'possession-002304.pt', run / 'possession-004864.pt']
    ds = [run / 'pressing-000000.pt', run / 'pressing-002560.pt', run / 'pressing-005120.pt']
    results['generation_matrix'] = matrix(ps, ds, Config(), seeds)
    # Share provenance without embedding this machine's absolute checkout path.
    for row in results['generation_matrix']['matchups']:
        row['possession'] = str(Path(row['possession']).relative_to(root))
        row['defending'] = str(Path(row['defending']).relative_to(root))
    (root / 'docs/V2_EVALUATION.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {m: v[m] for m in ('mean_passes', 'mean_duration', 'interceptions', 'success_rate')}
                      for k, v in results.items() if isinstance(v, dict) and 'mean_passes' in v}, indent=2))
