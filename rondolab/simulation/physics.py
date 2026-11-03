"""Reach estimates use current motion only. Collision tests use first relative-circle entry."""
import math

import numpy as np


def first_contact(ball_start, ball_end, player_start, player_end, radius):
    origin = ball_start - player_start
    delta = (ball_end - ball_start) - (player_end - player_start)
    cc = float(origin @ origin - radius * radius)
    if cc <= 0:
        return 0.
    aa = float(delta @ delta)
    bb = float(2 * origin @ delta)
    disc = bb * bb - 4 * aa * cc
    if aa < 1e-12 or disc < 0:
        return None
    t = (-bb - math.sqrt(disc)) / (2 * aa)
    return t if 0 <= t <= 1 else None


def reach_distance(player, target, seconds, config):
    delta = target - player.pos
    distance = float(np.linalg.norm(delta))
    heading = math.atan2(delta[1], delta[0])
    turn = abs(math.atan2(math.sin(heading - player.facing), math.cos(heading - player.facing)))
    available = max(0., seconds - config.reaction_time - turn / config.turning_rate * .35)
    initial = max(0., float(player.vel @ (delta / max(distance, 1e-9))))
    top = config.defender_speed
    initial = min(initial, top)
    accelerate = min(available, max(0., (top - initial) / config.acceleration))
    return initial * accelerate + .5 * config.acceleration * accelerate ** 2 + top * (available - accelerate)


def intercept_point(world, player):
    b, c = world.ball, world.config
    fallback = visible_receiving_point(world)
    if b.carrier is not None:
        return fallback, 0., False
    speed = float(np.linalg.norm(b.vel))
    direction = b.vel / max(speed, 1e-9)
    for seconds in np.linspace(.05, 1.6, 24):
        travel = max(0., speed * seconds - .5 * c.resistance * seconds * seconds)
        point = b.pos + direction * travel
        if not (0 <= point[0] <= c.width and 0 <= point[1] <= c.height):
            break
        distance = float(np.linalg.norm(point - player.pos))
        if distance <= reach_distance(player, point, seconds, c) + c.interception_radius:
            return point, float(seconds), True
    return fallback, .25, False


def visible_receiving_point(world):
    """Predict a receiver from the visible flight ray, never Ball.target or pending actions."""
    b, c = world.ball, world.config
    if b.carrier is not None:
        return b.pos.copy()
    speed = float(np.linalg.norm(b.vel))
    direction = b.vel / max(speed, 1e-9)
    candidates = []
    for p in world.players[:6]:
        along = float((p.pos - b.pos) @ direction)
        seconds = along / max(speed, 1e-9)
        if .02 < seconds < 2.5:
            point = p.pos + p.vel * min(seconds, .4)
            perpendicular = float(np.linalg.norm(point - b.pos - direction * float((point - b.pos) @ direction)))
            if perpendicular < c.control_radius + .6:
                candidates.append((seconds + perpendicular * .2, point))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return np.clip(b.pos + b.vel * .4, [.4, .4], [c.width - .4, c.height - .4])


def lane_margin(world, receiver):
    a, c = world.ball.pos, world.config
    delta = receiver.pos - a
    length = float(np.linalg.norm(delta))
    margins = []
    for d in world.players[6:]:
        t = float(np.clip((d.pos - a) @ delta / max(length * length, 1e-9), .05, 1))
        point = a + delta * t
        seconds = t * length / (c.ball_speed * c.pass_strength)
        margins.append(float(np.linalg.norm(point - d.pos)) - c.interception_radius -
                       reach_distance(d, point, seconds, c))
    return min(margins)
