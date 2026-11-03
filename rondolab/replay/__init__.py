import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from rondolab.environments.parallel import action_mask, observation, reward_components
from rondolab.policies.heuristics import scripted_actions
from rondolab.simulation.config import Config
from rondolab.simulation.world import World


class Replay(BaseModel):
    schema_version: int = Field(2, ge=1, le=2)
    episode_id: str
    seed: int
    config: Config
    policy: dict
    opponent: dict | None = None
    frames: list[dict] = Field(min_length=2, max_length=20000)
    events: list[dict]
    decisions: list[dict]
    metrics: dict

    @model_validator(mode='before')
    @classmethod
    def legacy_version(cls, data):
        if isinstance(data, dict) and data.get('schema_version', 1) == 1:
            config = data['config'].model_dump() if isinstance(data['config'], Config) else data['config']
            data = {**data, 'config': {**config, 'environment_version': 1}}
        return data

    @model_validator(mode='after')
    def frames_valid(self):
        previous = -1.
        import math
        for frame in self.frames:
            if not math.isfinite(frame['time']) or frame['time'] <= previous:
                raise ValueError('Replay timestamps must be finite and strictly increasing')
            previous = frame['time']
            if len(frame['players']) != 8:
                raise ValueError('Only 6v2 replay schemas are supported')
            points = [p['position'] for p in frame['players']] + [frame['ball']['position']]
            if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
                raise ValueError('Invalid replay positions')
        return self


def episode_metrics(world):
    events = world.events
    interceptions = [e for e in events if e['type'] in ('interception', 'tackle')]
    result = {'passes': world.passes, 'attempts': world.attempts,
            'completion': world.passes / world.attempts if world.attempts else 0,
            'duration': world.time, 'turnovers': int(world.done and world.reason != 'time_limit'),
            'out_of_bounds': int(world.reason == 'out_of_bounds'), 'reason': world.reason,
            'distinct_players': len(world.involved), 'receivers': world.receivers,
            'mean_hold': sum(world.hold_times) / len(world.hold_times) if world.hold_times else 0,
            'interceptions': len(interceptions), 'interception_locations': [e['position'] for e in interceptions],
            'time_to_interception': interceptions[0]['time'] if interceptions else None,
            'pressing_efficiency': len(interceptions) / max(world.time, .05)}
    if world.config.environment_version == 2:
        import math

        import numpy as np
        passes = [e for e in events if e['type'] == 'pass']
        receptions = [e for e in events if e['type'] == 'reception']
        s, count = world.statistics, max(1, world.statistics['samples'])
        flight_interceptions = [e for e in interceptions if e['type'] == 'interception']
        result.update(interceptions=len(flight_interceptions), tackles=sum(e['type'] == 'tackle' for e in interceptions),
                      defensive_contacts=len(interceptions), forced_turnovers=int(world.reason in ('interception', 'tackle')),
                      time_to_interception=flight_interceptions[0]['time'] if flight_interceptions else None,
                      time_to_regain=interceptions[0]['time'] if interceptions else None)
        pairs = {}
        passing_network = {}
        for e in passes:
            pair = '-'.join(map(str, sorted([e['player'], e['target']])))
            pairs[pair] = pairs.get(pair, 0) + 1
        for e in receptions:
            key = f'{e["origin"]}->{e["player"]}'
            passing_network[key] = passing_network.get(key, 0) + 1
        result.update(average_pass_distance=float(np.mean([e['distance'] for e in passes])) if passes else 0.,
                      average_ball_travel_time=float(np.mean([e['travel_time'] for e in receptions])) if receptions else 0.,
                      average_nearest_defender_distance=s['nearest_distance'] / count,
                      average_pressure=s['pressure'] / count, pressure_fraction=s['pressured'] / count,
                      average_carrier_pressure=s['carrier_pressure'] / max(1, s['carrier_samples']),
                      average_open_lanes=s['open_lanes'] * 5 / count,
                      both_chasing_fraction=s['both_chasing'] / count,
                      coordination_fraction=s['coordinated'] / count,
                      defender_movement_distance=s['defender_distance'],
                      pressured_receptions=s['pressured_receptions'], successful_split_passes=s['split_passes'],
                      central_interceptions=s['central_interceptions'],
                      pass_direction_bins=np.histogram([e['direction'] for e in passes],
                                                       bins=np.linspace(-math.pi, math.pi, 9))[0].tolist(),
                      pair_counts=pairs, passing_network=passing_network,
                      selected_receivers=[sum(e['target'] == i for e in passes) for i in range(6)],
                      adjacent_pass_fraction=sum((e['player'] - e['target']) % 6 in (1, 5)
                                                                   for e in passes) / max(1, len(passes)))
        from rondolab.diagnostics import flags
        result['flags'] = flags(result)
    return result


def generate_episode(config=None, seed=42, model=None, metadata=None, baseline='heuristic', opponent=None,
                     record=True, opponent_metadata=None):
    w = World(config, seed)
    metadata = metadata or {'name': 'Scripted · cover and press', 'team': 'none', 'steps': None,
                            'scenario': 'adaptive', 'kind': 'scripted'}
    frames, decisions = [w.frame()], []
    count = 0
    reward = {}
    while not w.done:
        if count % w.config.decision_steps == 0:
            w.step_events = []
            decision_observations = {i: observation(w, i).tolist() for i in range(8)} if record else {}
            decision_masks = {i: action_mask(w, i, (metadata or {}).get('scenario', 'adaptive')).tolist()
                              for i in range(8)} if record else {}
            actions = scripted_actions(w, simple=baseline == 'simple', random=baseline == 'random')
            for player in w.players:
                player.control, player.probabilities, player.value = 'scripted', None, None
            if baseline == 'random':
                for idx in range(6):
                    actions[idx] = int(w.rng.choice(action_mask(w, idx).nonzero()[0]))
            if model is not None:
                team = metadata['team']
                indices = [6, 7] if team == 'pressing' else list(range(6))
                if team == 'possession' and metadata.get('scenario') == 'basic':
                    indices = [w.ball.carrier] if w.ball.carrier is not None and w.ball.carrier < 6 else []
                if indices:
                    actions.update(model.decisions(w, indices, metadata.get('scenario', 'adaptive')))
            if opponent is not None:
                other = list(range(6)) if metadata['team'] == 'pressing' else [6, 7]
                if metadata['team'] == 'pressing' and (opponent_metadata or {}).get('scenario') == 'basic':
                    other = [w.ball.carrier] if w.ball.carrier is not None and w.ball.carrier < 6 else []
                if other:
                    actions.update(opponent.decisions(w, other, (opponent_metadata or {}).get('scenario', 'adaptive')))
            w.set_actions(actions)
            if record:
                decisions.append({'time': w.time, 'actions': actions,
                                  'observations': decision_observations, 'masks': decision_masks})
        w.tick()
        reward = reward_components(w, w.step_events)
        if record:
            frame = w.frame()
            frame['reward_components'] = reward
            frame['team_reward'] = sum(reward.values())
            frames.append(frame)
        count += 1
    metrics = episode_metrics(w)
    if not record:
        return metrics
    return Replay(schema_version=w.config.environment_version, episode_id=f'seed-{seed}', seed=seed,
                  config=w.config, policy=metadata, opponent=opponent_metadata,
                  frames=frames, events=w.events, decisions=decisions, metrics=metrics).model_dump()


def main():
    import argparse

    from rondolab.policies.neural import load_checkpoint
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--opponent-checkpoint')
    p.add_argument('--config')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='examples/replay.json')
    args = p.parse_args()
    model, metadata = load_checkpoint(args.checkpoint) if args.checkpoint else (None, None)
    opponent, opponent_metadata = load_checkpoint(args.opponent_checkpoint) if args.opponent_checkpoint else (None, None)
    if opponent and (not model or metadata['team'] == opponent_metadata['team'] or
                     metadata['environment_version'] != opponent_metadata['environment_version']):
        p.error('Replay pairs require opposing teams from the same environment version')
    config = Config(**(metadata['config'] | {'environment_version': metadata['environment_version']})) if metadata else Config()
    if args.config:
        config = Config.model_validate_json(Path(args.config).read_text())
    if model and model.environment_version != config.environment_version:
        p.error('Checkpoint and requested football environment versions differ')
    replay = generate_episode(config, args.seed, model, metadata, opponent=opponent,
                              opponent_metadata=opponent_metadata)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(replay))
    print(json.dumps(replay['metrics'], indent=2))
