import math

import numpy as np

from rondolab.policies.heuristics import scripted_actions
from rondolab.simulation.config import Config
from rondolab.simulation.world import World, angle_delta, norm, segment_distance


def test_seed_and_scripted_sequence():
    a, b = World(seed=17), World(seed=17)
    for _ in range(200):
        for w in (a, b):
            if round(w.time / w.config.dt) % 4 == 0:
                w.set_actions(scripted_actions(w))
            w.tick()
        assert a.frame() == b.frame()
        if a.done:
            break
    assert a.attempts > 0
    assert a.passes > 0


def test_ball_rolls_decelerates_without_teleport():
    w = World(Config(width=15, height=15, resistance=.5))
    w.ball.carrier = None
    w.ball.pos = np.array([7., 13.])
    w.ball.vel = np.array([3., 0.])
    old = w.ball.pos.copy()
    w.tick()
    assert np.isclose(w.ball.pos[0] - old[0], .15)
    assert np.isclose(norm(w.ball.vel), 2.975)


def test_swept_interception_and_boundaries():
    w = World(Config(dt=.05))
    w.ball.carrier = None
    w.ball.pos = np.array([5., 7.])
    w.ball.vel = np.array([18., 0.])
    w.players[6].pos = np.array([5.45, 7.])
    w.move_ball()
    assert w.reason == 'interception'
    assert w.ball.carrier == 6
    w = World()
    w.ball.carrier = None
    w.ball.pos = np.array([14.9, 7.])
    w.ball.vel = np.array([10., 0.])
    w.move_ball()
    assert w.reason == 'out_of_bounds'


def test_speed_turn_and_collision():
    w = World()
    p = w.players[6]
    p.facing = math.pi
    prev_angle = p.facing
    w.move_players()
    assert abs(angle_delta(p.facing, prev_angle)) <= w.config.turning_rate * w.config.dt + 1e-8
    for _ in range(100):
        w.tick()
        assert all(norm(p.vel) <= max(w.config.player_speed, w.config.defender_speed) + 1e-8 for p in w.players)
    w.players[0].pos = w.players[1].pos.copy()
    w.move_players()
    assert norm(w.players[0].pos - w.players[1].pos) >= .649
    assert segment_distance(np.array([.5, 0]), np.array([0., 0]), np.array([1., 0])) == (0, .5)


def test_pass_requires_real_travel():
    w = World(Config(defender_speed=0, pass_error=0))
    w.hold = .6
    direction = w.players[1].pos - w.players[0].pos
    w.players[0].facing = math.atan2(direction[1], direction[0])
    old = w.ball.pos.copy()
    w.release(1)
    assert np.array_equal(w.ball.pos, old)
    assert w.ball.carrier is None
    w.tick()
    assert 0 < norm(w.ball.pos - old) <= w.config.ball_speed * w.config.dt + 1e-8
    assert w.passes == 0
