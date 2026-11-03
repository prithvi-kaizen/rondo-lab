import math

import numpy as np
import pytest

from rondolab.environments.parallel import action_mask, observation, reward_components
from rondolab.matchups import matrix
from rondolab.policies.neural import ActorCritic, load_checkpoint, save_checkpoint
from rondolab.replay import Replay, generate_episode
from rondolab.rl.curriculum import promotion
from rondolab.rl.population import OpponentPool, fingerprint, frozen
from rondolab.simulation.config import Config
from rondolab.simulation.physics import first_contact, visible_receiving_point
from rondolab.simulation.world import World
from rondolab.train import train


def test_grid_support_is_controllable_and_bounded():
    w = World()
    assert w.config.width == 12
    assert len({round(p.anchor[1], 1) for p in w.players[:6]}) == 2
    initial = [p.pos.copy() for p in w.players[:6]]
    w.set_actions({**{i: 0 for i in range(6)}, 6: 10, 7: 10})
    for _ in range(5):
        w.tick()
    assert all(np.array_equal(p.pos, point) for p, point in zip(w.players[:6], initial, strict=True))
    w.set_actions({0: 0, 1: 7, 6: 10, 7: 10})
    old = w.players[1].pos.copy()
    for _ in range(10):
        previous = w.players[1].vel.copy()
        w.tick()
        assert np.linalg.norm(w.players[1].vel - previous) <= w.config.acceleration * w.config.dt + 1e-8
    assert w.players[1].pos[0] < old[0]
    lo, hi = w.regions[1]
    assert np.all(w.players[1].pos >= lo) and np.all(w.players[1].pos <= hi)


def test_relative_swept_contact_reachable_and_unreachable():
    a, b = np.array([0., 0.]), np.array([10., 0.])
    assert first_contact(a, b, np.array([5., -2.]), np.array([5., 2.]), .5) is not None
    assert first_contact(a, b, np.array([5., 2.]), np.array([5., 3.]), .5) is None
    w = World(Config(ball_speed=18))
    w.ball.carrier = None
    w.ball.pos = np.array([5., 6.])
    w.ball.vel = np.array([18., 0.])
    w.players[6].pos = np.array([5.5, 6.1])
    w.previous_positions[6] = w.players[6].pos.copy()
    w.move_ball()
    assert w.reason == 'interception'
    assert w.ball.pos[0] < 5.5  # First contact, not closest approach or endpoint.
    w = World(Config(ball_speed=18))
    w.ball.carrier = None
    w.ball.pos = np.array([5., 3.])
    w.ball.vel = np.array([18., 0.])
    w.move_ball()
    assert not w.done and w.ball.carrier is None


def test_reception_preparation_orientation_and_touch_limit():
    w = World(Config(defender_speed=0))
    w.hold = .1
    assert not w.release(1)
    w.hold = .6
    delta = w.players[1].pos - w.ball.pos
    w.players[0].facing = math.atan2(delta[1], delta[0]) + math.pi
    assert not w.release(1)
    w.players[0].facing -= math.pi
    assert w.release(1)
    assert w.ball.carrier is None
    held = World(Config(max_touches=1, defender_speed=0))
    for _ in range(18):
        held.tick()
    assert held.reason == 'holding_limit'


def test_observations_hide_opponent_intentions_and_model_masks():
    w = World()
    old = observation(w, 6).copy()
    w.pending_pass = 4
    w.actions[0] = 5
    np.testing.assert_array_equal(old, observation(w, 6))
    w.ball.carrier = None
    w.ball.vel = np.array([7., 0.])
    point = visible_receiving_point(w)
    w.ball.target = 5
    np.testing.assert_array_equal(point, visible_receiving_point(w))
    model = ActorCritic()
    acts = model.decisions(w, list(range(6)))
    assert len(acts) == 6
    for i, a in acts.items():
        assert action_mask(w, i)[a]
        assert np.isclose(sum(w.players[i].probabilities), 1)
    assert len(model.decisions(w, [6, 7])) == 2
    assert w.frame()['players'][6]['pressure'] < 1
    frozen_model = frozen(model)
    assert not any(p.requires_grad for p in frozen_model.parameters())


def test_rewards_do_not_reward_sideways_or_split_labels():
    w = World()
    neutral = reward_components(w, [{'type': 'reception', 'split': False}])
    split = reward_components(w, [{'type': 'reception', 'split': True}])
    assert neutral == split
    assert neutral['pass'] == .08
    assert reward_components(w, [{'type': 'interception'}])['turnover'] == -4
    w.config.reward = 'sparse'
    assert sum(reward_components(w, [{'type': 'reception'}]).values()) == 0


def test_pool_sampling_preserves_all_types_and_history(tmp_path):
    pool = OpponentPool()
    for i in range(4):
        pool.add('pressing', tmp_path / f'd{i}.pt', i, i * 32, {'mean_duration': 10 - i})
    rng = np.random.default_rng(10)
    ids = {pool.sample('pressing', rng)['id'] for _ in range(400)}
    assert 'scripted-cover' in ids and 'scripted-chase' in ids
    assert all(str((tmp_path / f'd{i}.pt').resolve()) in ids for i in range(4))
    with pytest.raises(ValueError, match='immutable'):
        pool.add('pressing', tmp_path / 'd0.pt', 99, 0)


def test_alternation_resume_and_evaluation_are_reproducible(tmp_path):
    kwargs = {'mode': 'both', 'scenario': 'adaptive', 'block_steps': 32, 'checkpoint_interval': 32,
              'num_envs': 2, 'evaluation_episodes': 1, 'seed': 2}
    uninterrupted = train(timesteps=128, output=tmp_path / 'full', **kwargs)
    first = train(timesteps=64, output=tmp_path / 'first', **kwargs)
    continued = train(timesteps=128, output=tmp_path / 'continued', resume=tmp_path / 'first/session.pt')
    assert [r['team'] for r in uninterrupted] == ['possession', 'pressing', 'possession', 'pressing']
    for row in continued:
        assert row['benchmark_config']['defender_strategy'] == 'cover_press'
        assert row['benchmark_config'] == continued[0]['benchmark_config']
        learning, other = row['team'], row['frozen_team']
        assert row['parameter_hashes_before'][other] == row['parameter_hashes_after'][other]
        assert row['parameter_hashes_before'][learning] != row['parameter_hashes_after'][learning]
    assert continued[-1]['generation'] == 2
    assert continued[-1]['parameter_hashes_after'] == uninterrupted[-1]['parameter_hashes_after']
    assert first[-1]['steps'] == 64
    p = tmp_path / 'full/possession-000096.pt'
    d = tmp_path / 'full/pressing-000128.pt'
    pm, _ = load_checkpoint(p)
    dm, _ = load_checkpoint(d)
    before = [fingerprint(pm), fingerprint(dm)]
    evaluated = matrix([p], [d], seeds=[10001, 10002])
    assert evaluated['matchups'][0]['seeds'] == [10001, 10002]
    assert before == [fingerprint(pm), fingerprint(dm)]


def test_legacy_checkpoint_and_replay_are_explicit(tmp_path):
    model = ActorCritic(90, 1)
    path = tmp_path / 'legacy.pt'
    config = Config(environment_version=1, width=15, height=15, ball_speed=10)
    save_checkpoint(path, model, {'team': 'possession', 'scenario': 'basic', 'config': config.model_dump()})
    loaded, meta = load_checkpoint(path)
    replay = generate_episode(config, 42, loaded, meta)
    assert replay['schema_version'] == 1
    assert len(replay['decisions'][0]['observations'][0]) == 90
    replay['config'].pop('environment_version')
    assert Replay.model_validate(replay).config.environment_version == 1
    with pytest.raises(ValueError, match='versions differ'):
        loaded.decisions(World(), [0])
    with pytest.raises(ValueError, match='session.pt'):
        train(resume=path, output=tmp_path / 'invalid')


def test_curriculum_requires_performance_and_sample_count():
    low_sample = {'episodes': 2, 'mean_passes': 30, 'mean_duration': 40}
    assert promotion(1, low_sample) == 1
    qualified = dict(low_sample, episodes=10)
    assert promotion(1, qualified) == 2
    assert promotion(1, dict(qualified, mean_passes=0)) == 1


def test_basic_passing_keeps_scripted_off_ball_movement():
    model = ActorCritic()
    replay = generate_episode(Config(), model=model, metadata={'team': 'possession', 'scenario': 'basic'})
    assert replay['frames'][1]['players'][0]['control'] == 'PPO'
    assert all(p['control'] == 'scripted' for p in replay['frames'][1]['players'][1:6])
    assert replay['frames'][1]['players'][1]['action'] == 9
    assert action_mask(World(), 1, 'basic')[9] == 1
