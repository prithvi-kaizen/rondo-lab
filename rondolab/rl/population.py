"""Immutable on-disk opponents. Sampling always leaves historical and scripted support."""
import copy
import hashlib
from pathlib import Path

from rondolab.policies.neural import load_checkpoint


def fingerprint(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def frozen(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


class OpponentPool:
    def __init__(self, entries=None, weights=(.4, .35, .25)):
        self.entries = copy.deepcopy(entries) if entries is not None else [
            {'id': 'scripted-chase', 'team': 'pressing', 'kind': 'scripted', 'strategy': 'chase', 'generation': 0},
            {'id': 'scripted-cover', 'team': 'pressing', 'kind': 'scripted', 'strategy': 'cover_press', 'generation': 0},
            {'id': 'scripted-possession', 'team': 'possession', 'kind': 'scripted', 'strategy': 'heuristic', 'generation': 0},
            {'id': 'random-possession', 'team': 'possession', 'kind': 'scripted', 'strategy': 'random', 'generation': 0},
        ]
        if len(weights) != 3 or any(w <= 0 for w in weights):
            raise ValueError('Latest, historical and scripted pool weights must all be positive')
        self.weights = list(weights)
        self._cache = {}

    def add(self, team, path, generation, steps, evaluation=None):
        identifier = str(Path(path).resolve())
        if any(e['id'] == identifier for e in self.entries):
            raise ValueError('Opponent snapshots are immutable and cannot be overwritten')
        self.entries.append({'id': identifier, 'path': identifier, 'team': team, 'kind': 'checkpoint',
                             'generation': generation, 'steps': steps, 'evaluation': evaluation})

    def sample(self, team, rng):
        entries = [e for e in self.entries if e['team'] == team]
        neural = [e for e in entries if e['kind'] == 'checkpoint']
        scripted = [e for e in entries if e['kind'] == 'scripted']
        buckets = [neural[-1:], neural, scripted]
        weights = [w if bucket else 0 for w, bucket in zip(self.weights, buckets, strict=True)]
        total = sum(weights)
        bucket = buckets[int(rng.choice(3, p=[w / total for w in weights]))]
        # Uniform historical sampling keeps early/weak opponents reachable. The best snapshot
        # receives extra probability, while every other historical snapshot retains support.
        if bucket is neural and len(neural) > 2 and rng.random() < .3:
            def score(entry):
                evaluation = entry.get('evaluation') or {}
                value = evaluation.get('mean_duration', 0)
                return value if team == 'possession' else -value
            evaluated = [e for e in neural if e.get('evaluation')]
            return copy.deepcopy(max(evaluated or neural, key=score))
        return copy.deepcopy(bucket[int(rng.integers(len(bucket)))])

    def model(self, entry, device='cpu'):
        if entry['kind'] == 'scripted':
            return None
        key = (entry['id'], device)
        if key not in self._cache:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            model, _ = load_checkpoint(entry['path'], device)
            self._cache[key] = model.eval().requires_grad_(False)
        return self._cache[key]
