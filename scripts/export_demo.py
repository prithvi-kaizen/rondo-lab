"""Publish recorded simulation states, never model weights or fabricated stages."""
import hashlib
import json
from pathlib import Path

from rondolab.policies.neural import load_checkpoint
from rondolab.replay import Replay, generate_episode
from rondolab.simulation.config import Config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'frontend/public/demo'


def export():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entries = []
    sources = [('scripted-42', 'Scripted · opening scenario', 42, None),
               ('scripted-44', 'Scripted · interception sequence', 44, None),
               ('scripted-10001', 'Scripted · held-out scenario', 10001, None),
               ('generation-0', 'Untrained · generation 0', 42, 'possession-000000.pt'),
               ('generation-5', 'Intermediate · generation 5', 42, 'possession-002304.pt'),
               ('generation-10', 'Final recorded · generation 10', 42, 'possession-004864.pt')]
    for identifier, label, seed, filename in sources:
        path = ROOT / 'experiments/v2-release' / filename if filename else None
        model, metadata = load_checkpoint(path) if path else (None, None)
        replay = generate_episode(Config(), seed, model, metadata)
        # Public playback needs provenance, not local filesystem paths or optimizer settings.
        replay['policy'] = {k: v for k, v in replay['policy'].items() if k in
                            ('kind', 'name', 'team', 'steps', 'stage', 'generation', 'environment_version')}
        Replay.model_validate(replay)
        data = json.dumps(replay, separators=(',', ':'))
        if str(ROOT) in data or '/Users/' in data:
            raise ValueError('Local path leaked into public replay')
        (OUTPUT / f'{identifier}.json').write_text(data)
        entries.append({'id': identifier, 'label': label, 'seed': seed, 'file': f'{identifier}.json',
                        'checkpoint': f'v2-release/{filename}' if filename else None,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
                        'steps': metadata['steps'] if metadata else 0,
                        'generation': metadata.get('generation') if metadata else None,
                        'duration': replay['metrics']['duration'], 'passes': replay['metrics']['passes'],
                        'reason': replay['metrics']['reason']})
    (OUTPUT / 'manifest.json').write_text(json.dumps({'schema': 1, 'recordings': entries}, indent=2))
    print(f'Exported {len(entries)} actual replay recordings to {OUTPUT}')


if __name__ == '__main__':
    export()
