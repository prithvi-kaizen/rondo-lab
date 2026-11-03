import math

import numpy as np

from rondolab.legacy.world import Ball, Player, angle_delta, norm, segment_distance, unit
from rondolab.legacy.world import World as LegacyWorld
from rondolab.simulation.physics import first_contact, intercept_point, lane_margin


class World(LegacyWorld):
    """Environment v2; v1 is an explicit frozen engine, never an implicit weight transfer."""
    def __new__(cls, config=None, seed=42):
        if config is not None and config.environment_version == 1:
            return LegacyWorld(config, seed)
        return super().__new__(cls)

    def reset(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        c = self.config
        self.players = []
        anchors = [(.8, .8), (c.width / 2, .8), (c.width - .8, .8),
                   (c.width - .8, c.height - .8), (c.width / 2, c.height - .8), (.8, c.height - .8)]
        center = np.array([c.width / 2, c.height / 2])
        self.regions = []
        for i, (x, y) in enumerate(anchors):
            anchor = np.array([x, y])
            p = anchor + self.rng.uniform(-.12, .12, 2)
            self.players.append(Player(i, 'possession', p, anchor,
                                       facing=math.atan2(*(center - p)[::-1])))
            reach = c.support_radius
            lower = np.maximum([.45, .45], anchor - [reach, reach])
            upper = np.minimum([c.width - .45, c.height - .45], anchor + [reach, reach])
            self.regions.append((lower, upper))
        shift = self.rng.uniform(-.7, .7, 2)
        for i, offset in enumerate(([-1.0, -.8], [1., 1.])):
            p = center + offset + shift + self.rng.uniform(-.2, .2, 2)
            self.players.append(Player(i + 6, 'defending', p, p.copy(), facing=math.pi))
        for p in self.players:
            p.previous_action = 0
            p.target = p.pos.copy()
            p.last_interception_attempt = None
        self.ball = Ball(self.players[0].pos.copy())
        self.time = self.hold = 0.
        self.done = False
        self.reason = None
        self.events, self.step_events = [], []
        self.passes = self.attempts = 0
        self.pair = None
        self.pair_count = 0
        self.actions = {i: 0 for i in range(8)}
        self.targets = {p.id: p.pos.copy() for p in self.players}
        self.reaction_targets = {p.id: p.pos.copy() for p in self.players}
        self.previous_positions = [p.pos.copy() for p in self.players]
        self.next_reaction = 0.
        self.receivers = [0] * 6
        self.hold_times = []
        self.involved = {0}
        self.pending_pass = None
        self.pending_since = 0.
        self.receive_delay = c.receiving_time
        self.flight_context = {}
        self.attempted_defenders = set()
        self.contact_cooldowns = {}
        self.statistics = {'samples': 0, 'nearest_distance': 0., 'pressure': 0., 'pressured': 0,
                           'open_lanes': 0., 'both_chasing': 0, 'coordinated': 0,
                           'defender_distance': [0., 0.], 'pressured_receptions': 0,
                           'split_passes': 0, 'central_interceptions': 0}
        return self

    def pressure(self, point):
        distance = min(norm(p.pos - point) for p in self.players[6:])
        return max(0., 1 - distance / self.config.pressure_radius)

    def project_region(self, idx, point):
        lower, upper = self.regions[idx]
        return np.clip(point, lower, upper)

    def set_actions(self, actions):
        self.actions = {int(i): int(a) for i, a in actions.items()}
        for p in self.players:
            p.previous_action, p.action = p.action, self.actions.get(p.id, 0)
        carrier = self.ball.carrier
        chosen = self.actions.get(carrier, 0)
        target = chosen - 1 if carrier is not None and carrier < 6 and 1 <= chosen <= 6 else None
        if target != self.pending_pass:
            self.pending_pass = target
            self.pending_since = self.time

    def release(self, target):
        b, c = self.ball, self.config
        if b.carrier is None or b.carrier >= 6 or target == b.carrier:
            return False
        p, receiver = self.players[b.carrier], self.players[target]
        if self.hold < self.receive_delay or norm(b.pos - p.pos) > .15:
            return False
        if self.pending_pass is not None and self.time - self.pending_since < c.pass_preparation:
            return False
        # Predict only visible velocity, bounded by the receiver's assigned support region.
        travel = norm(receiver.pos - b.pos) / (c.ball_speed * c.pass_strength)
        aim = self.project_region(target, receiver.pos + receiver.vel * min(travel, .45))
        direction = unit(aim - b.pos)
        turn = abs(angle_delta(math.atan2(direction[1], direction[0]), p.facing))
        if turn > .55:
            return False
        pair = tuple(sorted((p.id, target)))
        if c.pair_limit and pair == self.pair and self.pair_count >= c.pair_limit:
            self.event('illegal_pair', player=p.id)
            self.pending_pass = None
            return False
        self.pair_count = self.pair_count + 1 if pair == self.pair else 1
        self.pair = pair
        pressure = self.pressure(p.pos)
        error = self.rng.normal(0, c.pass_error * (1 + pressure * 2 + turn))
        co, si = math.cos(error), math.sin(error)
        direction = np.array([co * direction[0] - si * direction[1], si * direction[0] + co * direction[1]])
        delta = receiver.pos - b.pos
        signed = [float(delta[0] * (d.pos[1] - b.pos[1]) - delta[1] * (d.pos[0] - b.pos[0]))
                  for d in self.players[6:]]
        projections = [float((d.pos - b.pos) @ delta / max(delta @ delta, 1e-9)) for d in self.players[6:]]
        split = signed[0] * signed[1] < 0 and all(0 < t < 1 for t in projections) and all(
            segment_distance(d.pos, b.pos, receiver.pos)[0] < 2.5 for d in self.players[6:])
        self.flight_context = {'split': bool(split), 'player': p.id, 'target': target,
                               'release_time': self.time, 'pressure': pressure}
        b.origin = b.pos.tolist()
        b.previous_carrier, b.last_touch = p.id, p.id
        b.target, b.carrier, b.flight_time = target, None, 0.
        b.vel = direction * c.ball_speed * c.pass_strength
        self.hold_times.append(self.hold)
        self.attempts += 1
        self.hold = 0.
        self.pending_pass = None
        self.attempted_defenders.clear()
        self.event('pass', player=p.id, target=target, origin=b.origin, distance=norm(delta),
                   direction=math.atan2(direction[1], direction[0]), split=bool(split), pressure=pressure)
        return True

    def tactical_target(self, p):
        c, b, action = self.config, self.ball, self.actions.get(p.id, 0)
        if p.id < 6:
            nearest = min(self.players[6:], key=lambda d: norm(d.pos - p.pos))
            if action == 7:
                target, intention = p.pos + [-c.support_radius, 0], 'adjust left in region'
            elif action == 8:
                target, intention = p.pos + [c.support_radius, 0], 'adjust right in region'
            elif action == 10:
                target, intention = p.pos + unit(p.pos - nearest.pos) * c.support_radius, 'separate from pressure'
            elif action == 11:
                target, intention = p.anchor, 'recover anchor'
            elif action == 9:
                # Search within a bounded region for an observable receiving angle.
                lo, hi = self.regions[p.id]
                candidates = [p.anchor, lo, hi, np.array([lo[0], hi[1]]), np.array([hi[0], lo[1]])]
                def support_score(point):
                    clearance = min(segment_distance(d.pos, b.pos, point)[0] for d in self.players[6:])
                    separation = min(norm(d.pos - point) for d in self.players[6:])
                    spacing = min(norm(q.pos - point) for q in self.players[:6] if q.id != p.id)
                    return clearance + .35 * separation + .15 * min(spacing, 3) - .12 * norm(point - p.pos)
                target, intention = max(candidates, key=support_score), 'offer receiving angle'
            else:
                target, intention = p.pos, 'hold support position'
            if b.carrier == p.id:
                target = p.pos + unit(np.asarray(target) - p.pos) * min(.3, norm(np.asarray(target) - p.pos))
                intention = f'prepare pass to {self.pending_pass + 1}' if self.pending_pass is not None else 'control ball'
            p.intention = intention
            return self.project_region(p.id, target)
        p.intention = {0: 'angle press', 1: 'cover partner', 8: 'recover center', 9: 'attack observed ball path',
                       10: 'hold defensive shape', 11: 'close receiver'}.get(action, f'block lane to P{action - 1}')
        center = np.array([c.width / 2, c.height / 2])
        if action == 10:
            return p.pos.copy()
        if action == 8:
            return center + ([-.6, 0] if p.id == 6 else [.6, 0])
        if action == 9 and b.carrier is None:
            point, eta, feasible = intercept_point(self, p)
            if feasible and p.id not in self.attempted_defenders:
                self.attempted_defenders.add(p.id)
                p.last_interception_attempt = {'time': self.time, 'point': point.tolist(), 'eta': eta}
                self.event('interception_attempt', defender=p.id, position=point.tolist(), eta=eta)
            return point
        if action in (0, 9):
            # Offset the approach toward the center to close a lane on the way to the carrier.
            return b.pos + unit(center - b.pos) * min(.22, norm(center - b.pos))
        if 2 <= action <= 7:
            receiver = self.players[action - 2]
            _, t = segment_distance(p.pos, b.pos, receiver.pos)
            return b.pos + (receiver.pos - b.pos) * np.clip(t, .32, .78)
        other = self.players[13 - p.id]
        candidates = [q for q in self.players[:6] if q.id != b.carrier]
        receiver = min(candidates, key=lambda q: norm(q.pos - p.pos) - .3 * norm(q.pos - other.pos)
                       + .15 * norm(q.pos - b.pos))
        if action == 11:
            return receiver.pos + unit(b.pos - receiver.pos) * .7
        return b.pos * .35 + receiver.pos * .65

    def move_players(self):
        c, b = self.config, self.ball
        self.previous_positions = [p.pos.copy() for p in self.players]
        if self.time >= self.next_reaction:
            self.reaction_targets = {p.id: self.tactical_target(p).copy() for p in self.players}
            self.next_reaction = self.time + c.reaction_time
        for p in self.players:
            p.target = self.reaction_targets[p.id]
            delta = p.target - p.pos
            top = c.player_speed if p.id < 6 else c.defender_speed
            desired_heading = math.atan2(delta[1], delta[0]) if norm(delta) > .05 else p.facing
            if b.carrier == p.id and self.pending_pass is not None:
                aim = self.players[self.pending_pass].pos - p.pos
                desired_heading = math.atan2(aim[1], aim[0])
            elif p.id < 6 and norm(delta) < .2:
                aim = b.pos - p.pos
                if norm(aim) > .1:
                    desired_heading = math.atan2(aim[1], aim[0])
            turn = float(np.clip(angle_delta(desired_heading, p.facing), -c.turning_rate * c.dt,
                                 c.turning_rate * c.dt))
            p.angular_velocity = turn / c.dt
            p.facing += turn
            heading_error = abs(angle_delta(math.atan2(delta[1], delta[0]), p.facing)) if norm(delta) > .05 else 0
            desired = unit(delta) * min(top, norm(delta) * 2.5) * max(.2, math.cos(heading_error))
            change = desired - p.vel
            p.vel += unit(change) * min(norm(change), c.acceleration * c.dt)
            p.pos += p.vel * c.dt
            if p.id < 6:
                p.pos = self.project_region(p.id, p.pos)
            else:
                p.pos = np.clip(p.pos, [.35, .35], [c.width - .35, c.height - .35])
        for _ in range(2):
            for i, p in enumerate(self.players):
                for q in self.players[i + 1:]:
                    delta = p.pos - q.pos
                    d = norm(delta)
                    if d < .64:
                        correction = unit(delta if d > 1e-8 else np.array([1., 0.])) * (.64 - d) / 2
                        p.pos += correction
                        q.pos -= correction
        for p in self.players:
            if p.id < 6:
                p.pos = self.project_region(p.id, p.pos)

    def move_ball(self):
        b, c = self.ball, self.config
        if b.carrier is not None:
            p = self.players[b.carrier]
            delta = p.pos - b.pos
            movement = unit(delta) * min(norm(delta), c.ball_speed * c.pass_strength * c.dt)
            b.pos += movement
            b.vel = movement / c.dt
            if p.id < 6:
                self.hold += c.dt
                for d in self.players[6:]:
                    contact = first_contact(b.pos - movement, b.pos, self.previous_positions[d.id], d.pos,
                                            c.interception_radius)
                    if contact is not None:
                        self.turnover('tackle', d.id)
                        return
                limit = min(c.max_hold, {0: c.max_hold, 1: .85, 2: 1.5}[c.max_touches])
                if self.hold > limit:
                    self.turnover('holding_limit')
            return
        old, new = b.pos.copy(), b.pos + b.vel * c.dt
        b.flight_time += c.dt
        hits = []
        for p in self.players:
            if p.id == b.previous_carrier and b.flight_time < .25:
                continue
            if self.time < self.contact_cooldowns.get(p.id, -1):
                continue
            radius = c.interception_radius if p.id >= 6 else c.control_radius
            t = first_contact(old, new, self.previous_positions[p.id], p.pos, radius)
            if t is not None:
                hits.append((t, p.id))
        if hits:
            t, idx = min(hits)
            b.pos = old + (new - old) * t
            if idx >= 6:
                self.statistics['central_interceptions'] += int(.25 * c.width < b.pos[0] < .75 * c.width and
                                                                 .25 * c.height < b.pos[1] < .75 * c.height)
                self.turnover('interception', idx)
                return
            p = self.players[idx]
            incoming = math.atan2(-b.vel[1], -b.vel[0])
            orientation = abs(angle_delta(incoming, p.facing)) / math.pi
            pressure = self.pressure(p.pos)
            difficulty = .015 + .045 * orientation + .07 * pressure + .01 * max(0, norm(b.vel) - 9)
            if b.target is not None and self.rng.random() < difficulty:
                b.vel *= .5
                b.target = None
                self.contact_cooldowns[idx] = self.time + .18
                self.event('failed_reception', player=idx, pressure=pressure)
                return
            if b.target is not None:
                self.passes += 1
                self.receivers[idx] += 1
                self.involved.add(idx)
                self.statistics['pressured_receptions'] += int(pressure > 0)
                self.statistics['split_passes'] += int(self.flight_context.get('split', False))
                self.event('reception', player=idx, origin=b.previous_carrier, pressure=pressure,
                           split=self.flight_context.get('split', False),
                           travel_time=b.flight_time)
            else:
                self.event('recovery', player=idx)
            b.carrier, b.last_touch, b.target = idx, idx, None
            b.vel = p.vel.copy()
            self.receive_delay = c.receiving_time * (1 + .7 * orientation + .4 * pressure)
            self.pending_pass = None
            self.hold = 0.
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
        if self.pending_pass is not None:
            self.release(self.pending_pass)
        self.move_ball()
        self.time = round(self.time + self.config.dt, 8)
        s = self.statistics
        for j in range(2):
            s['defender_distance'][j] += norm(self.players[j + 6].pos - self.previous_positions[j + 6])
        if round(self.time / self.config.dt) % self.config.decision_steps == 0:
            distance = min(norm(d.pos - self.ball.pos) for d in self.players[6:])
            s['samples'] += 1
            s['nearest_distance'] += distance
            s['pressure'] += self.pressure(self.ball.pos)
            s['pressured'] += int(distance < self.config.pressure_radius)
            s['open_lanes'] += sum(lane_margin(self, p) > 0 for p in self.players[:6]
                                   if p.id != self.ball.carrier) / (5 if self.ball.carrier is not None else 6)
            acts = [self.actions.get(i, 0) for i in (6, 7)]
            s['both_chasing'] += int(acts == [0, 0])
            s['coordinated'] += int((0 in acts or 9 in acts) and any(a in range(1, 9) for a in acts)
                                    and norm(self.players[6].pos - self.players[7].pos) > 1.2)
        if not self.done and self.time >= self.config.duration:
            self.done, self.reason = True, 'time_limit'
            self.event('time_limit')

    def frame(self):
        frame = super().frame()
        for p, stored in zip(self.players, frame['players'], strict=True):
            stored.update(previous_action=p.previous_action, target=p.target.tolist(),
                          pressure=self.pressure(p.pos), ball_distance=norm(p.pos - self.ball.pos),
                          last_interception_attempt=p.last_interception_attempt,
                          region=[v.tolist() for v in self.regions[p.id]] if p.id < 6 else None,
                          receiving_remaining=max(0., self.receive_delay - self.hold)
                          if self.ball.carrier == p.id else 0.)
            if p.id >= 6 and self.ball.carrier is None:
                point, eta, feasible = intercept_point(self, p)
                stored['intercept_path'] = {'point': point.tolist(), 'eta': eta, 'feasible': feasible}
        return frame
