import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config


def norm(v):
    return float(np.linalg.norm(v))


def unit(v):
    return v / max(norm(v), 1e-9)


def angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def segment_distance(p, a, b):
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-10), 0, 1))
    return norm(p - a - t * d), t


@dataclass
class Player:
    id: int
    team: str
    pos: np.ndarray
    anchor: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    facing: float = 0
    angular_velocity: float = 0
    intention: str = 'support'
    action: int = 0
    probabilities: list | None = None
    value: float | None = None
    control: str = 'scripted'


@dataclass
class Ball:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    carrier: int | None = 0
    previous_carrier: int | None = None
    origin: list | None = None
    target: int | None = None
    last_touch: int = 0
    flight_time: float = 0


class World:
    def __init__(self, config=None, seed=42):
        self.config = config or Config()
        self.reset(seed)

    def reset(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        c = self.config
        center = np.array([c.width / 2, c.height / 2])
        self.players = []
        for i in range(6):
            a = math.pi + i * math.pi / 3
            p = center + np.array([math.cos(a) * (c.width / 2 - 1.5),
                                   math.sin(a) * (c.height / 2 - 1.4)])
            p += self.rng.uniform(-.18, .18, 2)
            self.players.append(Player(i, 'possession', p, p.copy(), facing=math.atan2(*(center - p)[::-1])))
        for i, offset in enumerate(([-1.3, -.8], [1.1, 1.2])):
            p = center + offset + self.rng.uniform(-.15, .15, 2)
            self.players.append(Player(i + 6, 'defending', p, p.copy(), facing=math.pi))
        self.ball = Ball(self.players[0].pos.copy())
        self.time = self.hold = 0.
        self.done = False
        self.reason = None
        self.events = []
        self.step_events = []
        self.passes = self.attempts = 0
        self.pair = None
        self.pair_count = 0
        self.actions = {i: 0 for i in range(8)}
        self.targets = {p.id: p.pos.copy() for p in self.players}
        self.reaction_targets = {p.id: p.pos.copy() for p in self.players}
        self.next_reaction = 0.
        self.receivers = [0] * 6
        self.hold_times = []
        self.involved = {0}
        return self

    def event(self, kind, **kwargs):
        e = {'type': kind, 'time': round(self.time, 4), **kwargs}
        self.events.append(e)
        self.step_events.append(e)

    def turnover(self, reason, defender=None):
        self.done, self.reason = True, reason
        self.event(reason, defender=defender, position=self.ball.pos.tolist())
        if defender is not None:
            self.ball.carrier = defender
            self.ball.last_touch = defender
        self.ball.vel[:] = 0

    def release(self, target):
        b, c = self.ball, self.config
        if b.carrier is None or b.carrier >= 6 or target == b.carrier:
            return
        p = self.players[b.carrier]
        if self.hold < .18:
            return
        pair = tuple(sorted((p.id, target)))
        if c.pair_limit and self.pair == pair and self.pair_count >= c.pair_limit:
            self.event('illegal_pair', player=p.id)
            return
        self.pair_count = self.pair_count + 1 if self.pair == pair else 1
        self.pair = pair
        receiver = self.players[target]
        # Lead the moving receiver using an estimate from currently visible velocity.
        travel_time = norm(receiver.pos - b.pos) / c.ball_speed
        support_target = self.tactical_target(receiver)
        lead = receiver.vel * min(travel_time, .6)
        remaining = norm(support_target - receiver.pos)
        aim = receiver.pos + unit(lead) * min(norm(lead), remaining)
        direction = unit(aim - b.pos)
        turn = abs(angle_delta(math.atan2(direction[1], direction[0]), p.facing))
        if turn > 1.25 and self.hold < .45:
            return
        pressure = max(0, 1 - min(norm(d.pos - p.pos) for d in self.players[6:]) / 2.5)
        error = self.rng.normal(0, c.pass_error * (1 + pressure + turn))
        co, si = math.cos(error), math.sin(error)
        direction = np.array([co * direction[0] - si * direction[1], si * direction[0] + co * direction[1]])
        b.origin = b.pos.tolist()
        b.previous_carrier, b.last_touch = p.id, p.id
        b.target, b.carrier, b.flight_time = target, None, 0
        b.vel = direction * c.ball_speed
        self.attempts += 1
        self.hold_times.append(self.hold)
        self.hold = 0
        self.event('pass', player=p.id, target=target, origin=b.origin)

    def set_actions(self, actions):
        self.actions = {int(i): int(a) for i, a in actions.items()}
        for p in self.players:
            p.action = self.actions.get(p.id, 0)
        b = self.ball
        if b.carrier is not None and b.carrier < 6:
            a = self.actions.get(b.carrier, 0)
            if 1 <= a <= 6:
                self.release(a - 1)

    def tactical_target(self, p):
        c, b, a = self.config, self.ball, self.actions.get(p.id, 0)
        if p.id < 6:
            pressure = min(self.players[6:], key=lambda d: norm(d.pos - p.pos))
            away = unit(p.pos - pressure.pos)
            toward = unit(b.pos - p.anchor)
            tangent = np.array([-toward[1], toward[0]])
            offset = (away * .7 + tangent * .25) * c.support_radius
            if a == 7:
                offset = tangent * c.support_radius
            elif a == 8:
                offset = -tangent * c.support_radius
            elif a == 9:
                offset = unit(b.pos - p.pos) * c.support_radius
            elif a == 10:
                offset = away * c.support_radius
            elif a == 11:
                offset = np.zeros(2)
            p.intention = {7: 'rotate clockwise', 8: 'rotate counterclockwise', 9: 'support angle',
                           10: 'away from pressure', 11: 'recover anchor'}.get(a, 'support')
            if b.carrier == p.id:
                p.intention = 'hold' if a == 0 else f'pass to {a}'
                offset *= .35
            target = p.anchor + offset
            return np.clip(target, [.65, .65], [c.width - .65, c.height - .65])
        p.intention = {0: 'press', 1: 'cover', 8: 'recover', 9: 'pursue ball'}.get(a, f'block {a - 1}')
        if a in (0, 9):
            return b.pos + b.vel * .1
        center = np.array([c.width / 2, c.height / 2])
        if a == 8:
            return center
        if 2 <= a <= 7:
            return b.pos * .48 + self.players[a - 2].pos * .52
        other = self.players[13 - p.id]
        options = [r for r in self.players[:6] if r.id != b.carrier]
        receiver = min(options, key=lambda r: norm(r.pos - other.pos) + .4 * norm(r.pos - b.pos))
        return b.pos * .4 + receiver.pos * .6

    def move_players(self):
        c = self.config
        if self.time >= self.next_reaction:
            self.reaction_targets = {p.id: self.tactical_target(p).copy() for p in self.players}
            self.next_reaction = self.time + c.reaction_time
        for p in self.players:
            delta = self.reaction_targets[p.id] - p.pos
            speed = c.player_speed if p.id < 6 else c.defender_speed
            if norm(delta) > .05:
                heading = math.atan2(delta[1], delta[0])
                turn = float(np.clip(angle_delta(heading, p.facing), -c.turning_rate * c.dt, c.turning_rate * c.dt))
                p.angular_velocity = turn / c.dt
                p.facing += turn
                alignment = max(.18, math.cos(angle_delta(heading, p.facing)))
                desired = unit(delta) * min(speed, norm(delta) * 2) * alignment
            else:
                desired = np.zeros(2)
            dv = desired - p.vel
            p.vel += unit(dv) * min(norm(dv), c.acceleration * c.dt)
            p.pos += p.vel * c.dt
            p.pos = np.clip(p.pos, [.35, .35], [c.width - .35, c.height - .35])
            if p.id < 6:
                offset = p.pos - p.anchor
                if norm(offset) > c.support_radius + .15:
                    p.pos = p.anchor + unit(offset) * (c.support_radius + .15)
                    p.vel *= .5
        # Equal positional correction resolves overlap without adding kinetic energy.
        for i, p in enumerate(self.players):
            for q in self.players[i + 1:]:
                delta = p.pos - q.pos
                distance = norm(delta)
                if distance < .65:
                    correction = unit(delta if distance > 1e-8 else np.array([1., 0.])) * (.65 - distance) / 2
                    p.pos += correction
                    q.pos -= correction

    def move_ball(self):
        b, c = self.ball, self.config
        if b.carrier is not None:
            p = self.players[b.carrier]
            # Feet are represented at the player center; no release-position jump.
            delta = p.pos - b.pos
            movement = unit(delta) * min(norm(delta), c.ball_speed * c.dt)
            b.pos += movement
            b.vel = movement / c.dt
            if p.id < 6:
                self.hold += c.dt
                if any(norm(d.pos - b.pos) <= c.interception_radius for d in self.players[6:]):
                    d = min(self.players[6:], key=lambda d: norm(d.pos - b.pos))
                    self.turnover('tackle', d.id)
                elif self.hold > min(c.max_hold, {0: c.max_hold, 1: .7, 2: 1.2}[c.max_touches]):
                    self.turnover('holding_limit')
            return
        # Swept tests ordered by path contact, independent of drawing frame rate.
        old = b.pos.copy()
        new = old + b.vel * c.dt
        b.flight_time += c.dt
        hits = []
        for p in self.players:
            if p.id == b.previous_carrier and b.flight_time < .25:
                continue
            radius = c.interception_radius if p.id >= 6 else c.control_radius
            distance, t = segment_distance(p.pos, old, new)
            if distance <= radius:
                hits.append((t, p.id))
        if hits:
            t, idx = min(hits)
            b.pos = old + (new - old) * t
            if idx >= 6:
                self.turnover('interception', idx)
                return
            p = self.players[idx]
            incoming = math.atan2(-b.vel[1], -b.vel[0])
            difficulty = abs(angle_delta(incoming, p.facing)) / math.pi
            # A poor first touch leaves a slower loose ball; never instant relocation.
            if idx == b.target and self.rng.random() < difficulty * .04:
                b.vel *= .45
                b.target = None
                self.event('failed_reception', player=idx)
            else:
                if b.target is not None:
                    self.passes += 1
                    self.receivers[idx] += 1
                    self.involved.add(idx)
                    self.event('reception', player=idx, origin=b.previous_carrier)
                else:
                    self.event('recovery', player=idx)
                # Receive at the swept contact point. Smoothly converge to feet through motion.
                b.carrier = idx
                b.last_touch = idx
                b.vel = p.vel.copy()
                b.target = None
                self.hold = 0
                return
        b.pos = new
        speed = norm(b.vel)
        b.vel = unit(b.vel) * max(0, speed - c.resistance * c.dt)
        if not (0 <= b.pos[0] <= c.width and 0 <= b.pos[1] <= c.height):
            self.turnover('out_of_bounds')
        elif speed < .2 and b.target is not None:
            b.target = None
            self.event('loose_ball')

    def tick(self):
        if self.done:
            return
        self.move_players()
        self.move_ball()
        self.time = round(self.time + self.config.dt, 8)
        if not self.done and self.time >= self.config.duration:
            self.done, self.reason = True, 'time_limit'
            self.event('time_limit')

    def frame(self):
        b = self.ball
        return {'time': self.time, 'players': [
            {'id': p.id, 'team': p.team, 'position': p.pos.tolist(), 'velocity': p.vel.tolist(),
             'facing': p.facing, 'angular_velocity': p.angular_velocity, 'action': p.action,
             'intention': p.intention, 'control': p.control, 'probabilities': p.probabilities,
             'value': p.value} for p in self.players],
            'ball': {'position': b.pos.tolist(), 'velocity': b.vel.tolist(), 'carrier': b.carrier,
                     'previous_carrier': b.previous_carrier, 'target': b.target, 'origin': b.origin,
                     'last_touch': b.last_touch, 'in_flight': b.carrier is None},
            'passes': self.passes, 'attempts': self.attempts, 'hold': self.hold}
