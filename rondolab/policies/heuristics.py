from rondolab.simulation.world import norm, segment_distance


def scripted_actions(world, simple=False, random=False):
    actions = {p.id: 0 for p in world.players}
    b, c = world.ball, world.config
    if b.carrier is not None and b.carrier < 6 and world.hold > (.22 if c.max_touches == 1 else .5):
        carrier = world.players[b.carrier]
        candidates = [p for p in world.players[:6] if p.id != carrier.id]
        if random:
            receiver = world.rng.choice(candidates)
        else:
            def score(p):
                lane = min(segment_distance(d.pos, b.pos, p.pos)[0] for d in world.players[6:])
                pressure = min(norm(d.pos - p.pos) for d in world.players[6:])
                distance = norm(p.pos - b.pos)
                return (pressure if simple else lane * 2 + pressure * .5) - distance * .12
            receiver = max(candidates, key=score)
        actions[b.carrier] = receiver.id + 1
    nearest = min(world.players[6:], key=lambda d: norm(d.pos - b.pos)).id
    for d in world.players[6:]:
        if c.defender_strategy == 'chase':
            actions[d.id] = 0
        elif c.defender_strategy == 'nearest':
            actions[d.id] = 0 if d.id == nearest else 8
        elif c.defender_strategy == 'lanes':
            options = [p for p in world.players[:6] if p.id != b.carrier]
            receiver = min(options, key=lambda p: norm(p.pos - d.pos))
            actions[d.id] = receiver.id + 2
        else:
            actions[d.id] = 0 if d.id == nearest else 1
        if b.carrier is None:
            actions[d.id] = 9 if d.id == nearest else actions[d.id]
    return actions
