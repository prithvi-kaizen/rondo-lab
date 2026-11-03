"""Football diagnostics from recorded states; flags request inspection, not strategy bans."""
import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from rondolab.simulation.physics import first_contact
from rondolab.simulation.world import norm, segment_distance


def reach_time(distance, speed, acceleration, reaction=0.):
    if distance <= 0:
        return reaction
    if speed <= 0:
        return 1e6
    acceleration_distance = speed * speed / (2 * acceleration)
    return reaction + (math.sqrt(2 * distance / acceleration) if distance <= acceleration_distance
                       else speed / acceleration + (distance - acceleration_distance) / speed)


def analyse_replay(replay):
    c, frames, events = replay['config'], replay['frames'], replay['events']
    passes = [e for e in events if e['type'] == 'pass']
    receptions = [e for e in events if e['type'] == 'reception']
    angles, distances, travel, adjacent, pairs, receivers = [], [], [], [], {}, [0] * 6
    reach = []
    for e in passes:
        f = min(frames, key=lambda f: abs(f['time'] - e['time']))
        origin = np.array(e['origin'])
        target = np.array(f['players'][e['target']]['position'])
        delta = target - origin
        distances.append(norm(delta))
        angles.append(math.atan2(delta[1], delta[0]))
        adjacent.append((e['player'] - e['target']) % 6 in (1, 5))
        key = '-'.join(map(str, sorted((e['player'], e['target']))))
        pairs[key] = pairs.get(key, 0) + 1
        receivers[e['target']] += 1
        outcome = next((r for r in events if r['time'] > e['time'] and
                        r['type'] in ('pass', 'reception', 'interception', 'out_of_bounds', 'failed_reception')), None)
        if outcome:
            travel.append(outcome['time'] - e['time'])
        times = []
        for d in f['players'][6:]:
            distance, _ = segment_distance(np.array(d['position']), origin, target)
            times.append(reach_time(max(0, distance - c['interception_radius']), c['defender_speed'],
                                    c['acceleration'], c['reaction_time']))
        reach.append(min(times))
    samples = frames[::max(1, c['decision_steps'])]
    nearest, lanes, pressures, coordination = [], [], [], []
    movement = np.zeros(2)
    tunnelling = 0
    for a, b in pairwise(frames):
        movement += [norm(np.array(b['players'][i]['position']) - a['players'][i]['position']) for i in (6, 7)]
        if c.get('environment_version', 1) == 2 and a['ball']['carrier'] is None and b['ball']['carrier'] is None:
            for i in (6, 7):
                tunnelling += int(first_contact(np.array(a['ball']['position']), np.array(b['ball']['position']),
                                               np.array(a['players'][i]['position']), np.array(b['players'][i]['position']),
                                               c['interception_radius']) is not None)
    for f in samples:
        ball = np.array(f['ball']['position'])
        ds = f['players'][6:]
        nearest.append(min(norm(np.array(d['position']) - ball) for d in ds))
        pressures.append(max(0, 1 - nearest[-1] / 2.5))
        if f['ball']['carrier'] is not None and f['ball']['carrier'] < 6:
            lanes.append(sum(min(segment_distance(np.array(d['position']), ball,
                            np.array(p['position']))[0] for d in ds) > c['interception_radius']
                             for p in f['players'][:6] if p['id'] != f['ball']['carrier']))
        coordination.append(int(ds[0]['action'] == ds[1]['action'] == 0))
    bins = np.histogram(angles, bins=np.linspace(-math.pi, math.pi, 9))[0].tolist()
    result = {'passes': len(receptions), 'interceptions': sum(e['type'] == 'interception' for e in events),
              'duration': frames[-1]['time'], 'reason': replay['metrics']['reason'],
              'average_pass_distance': float(np.mean(distances)) if distances else 0.,
              'average_ball_travel_time': float(np.mean(travel)) if travel else 0.,
              'average_nearest_defender_distance': float(np.mean(nearest)),
              'minimum_lane_reach_time': float(np.mean(reach)) if reach else None,
              'pass_direction_bins': bins, 'receiver_selection': receivers,
              'adjacent_pass_fraction': float(np.mean(adjacent)) if adjacent else 0.,
              'pair_counts': pairs, 'average_pressure': float(np.mean(pressures)),
              'pressure_fraction': float(np.mean(np.array(nearest) < 2.5)),
              'average_open_lanes': float(np.mean(lanes)) if lanes else 0.,
              'both_chasing_fraction': float(np.mean(coordination)),
              'defender_movement_distance': movement.tolist(),
              'reaction_time': c['reaction_time'], 'defender_speed': c['defender_speed'],
              'acceleration': c['acceleration'], 'ball_speed': c['ball_speed'], 'tunnelling_candidates': tunnelling}
    result['flags'] = flags(result)
    return result


def flags(metrics):
    result = []
    if metrics.get('reason') == 'holding_limit' and metrics.get('passes', 0) == 0:
        result.append('No pass before a holding-limit turnover; inspect stalling or preparation churn.')
    if metrics.get('reason') == 'out_of_bounds' and metrics.get('passes', 0) == 0 and metrics.get('duration', 10) < 2:
        result.append('Early boundary exit without a completed pass; inspect termination-seeking behavior.')
    if metrics.get('tunnelling_candidates', 0):
        result.append('A ball segment overlaps defensive reach while remaining in flight; inspect collision tunneling.')
    bins = metrics.get('pass_direction_bins', [])
    if sum(bins) >= 5 and max(bins) / sum(bins) > .7:
        result.append('Direction concentration above 70%; inspect repeated circulation.')
    pairs = metrics.get('pair_counts', {})
    if sum(pairs.values()) >= 5 and max(pairs.values()) / sum(pairs.values()) > .8:
        result.append('One pair accounts for over 80% of passes.')
    if metrics.get('pressure_fraction', 1) < .1:
        result.append('Defenders pressure the ball on fewer than 10% of sampled frames.')
    if metrics.get('both_chasing_fraction', 0) > .8:
        result.append('Both defenders chase together on over 80% of decisions.')
    if metrics.get('duration', 0) >= 25 and metrics.get('interceptions', 0) == 0:
        result.append('Long uninterrupted possession without an interception; inspect pressure.')
    return result


def main():
    from rondolab.policies.neural import load_checkpoint
    from rondolab.replay import generate_episode
    from rondolab.simulation.config import Config
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--output', default='docs/v2-baseline-diagnostics.json')
    args = p.parse_args()
    model, meta = load_checkpoint(args.checkpoint) if args.checkpoint else (None, None)
    config = Config(**(meta['config'] | {'environment_version': meta['environment_version']})) if meta else Config()
    results = [analyse_replay(generate_episode(config, seed, model, meta)) for seed in range(42, 42 + args.episodes)]
    Path(args.output).write_text(json.dumps({'checkpoint': args.checkpoint, 'episodes': results,
                                           'definitions': 'Direction bins span -pi to pi. Lane reach time is an '
                                           'optimistic from-rest reach estimate, not observed intent.'}, indent=2))
    print(json.dumps({key: float(np.mean([r[key] for r in results])) for key in [
        'passes', 'interceptions', 'duration', 'average_pass_distance', 'average_ball_travel_time',
        'average_nearest_defender_distance', 'adjacent_pass_fraction', 'average_pressure', 'average_open_lanes']}, indent=2))


if __name__ == '__main__':
    main()
