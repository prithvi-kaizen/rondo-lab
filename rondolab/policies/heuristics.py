from rondolab.legacy.heuristics import scripted_actions as legacy_actions
from rondolab.simulation.physics import intercept_point, lane_margin
from rondolab.simulation.world import norm


def scripted_actions(world, simple=False, random=False):
    if world.config.environment_version == 1:
        return legacy_actions(world, simple, random)
    actions = {p.id: (11 if simple else 9) for p in world.players}
    b, c = world.ball, world.config
    carrier = b.carrier
    if carrier is not None and carrier < 6:
        actions[carrier] = 0
        release_after = max(world.receive_delay, .22 if c.max_touches == 1 else .35)
        if world.hold >= release_after:
            candidates = [p for p in world.players[:6] if p.id != carrier]
            if c.pair_limit and world.pair and world.pair_count >= c.pair_limit:
                candidates = [p for p in candidates if tuple(sorted((carrier, p.id))) != world.pair]
            if candidates:
                def score(p):
                    safety = lane_margin(world, p)
                    space = min(norm(p.pos - d.pos) for d in world.players[6:])
                    return space if simple else safety + .3 * space
                receiver = world.rng.choice(candidates) if random else max(candidates, key=score)
                actions[carrier] = receiver.id + 1
    defenders = world.players[6:]
    if b.carrier is None:
        choices = [(intercept_point(world, d), d) for d in defenders]
        feasible = [(point[1], norm(d.pos - point[0]), d.id) for point, d in choices if point[2]]
        presser = min(feasible)[2] if feasible else min(defenders, key=lambda d: norm(d.pos - b.pos)).id
    else:
        presser = min(defenders, key=lambda d: norm(d.pos - b.pos) + .15 * norm(d.vel)).id
    candidates = [p for p in world.players[:6] if p.id != carrier]
    outlets = sorted(candidates, key=lambda p: lane_margin(world, p), reverse=True)[:3]
    for d in defenders:
        if c.defender_strategy == 'chase':
            actions[d.id] = 9 if b.carrier is None else 0
        elif c.defender_strategy == 'nearest':
            actions[d.id] = (9 if b.carrier is None else 0) if d.id == presser else 8
        elif d.id == presser and c.defender_strategy != 'lanes':
            actions[d.id] = 9 if b.carrier is None else 0
        else:
            # Cover likely exits on the opposite side of the press, using only observable geometry.
            other = world.players[13 - d.id]
            receiver = min(outlets, key=lambda p: norm(d.pos - (b.pos * .35 + p.pos * .65))
                           - .3 * norm(other.pos - p.pos))
            actions[d.id] = receiver.id + 2
    return actions
