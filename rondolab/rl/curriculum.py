from rondolab.simulation.config import Config

LEVELS = {1: (0, 'chase'), 2: (1, 'chase'), 3: (1.8, 'nearest'), 4: (2.3, 'lanes'),
          5: (2.8, 'cover_press'), 6: (3.2, 'cover_press')}


def level_config(level, base=None):
    c = (base or Config()).model_dump()
    speed, strategy = LEVELS[level]
    return Config(**(c | {'defender_speed': speed, 'defender_strategy': strategy, 'curriculum': level}))


def promotion(level, evaluation, threshold=8, minimum_episodes=10):
    """Explicit recommendation only; no automatic time-based difficulty increases."""
    return min(level + 1, 6) if (evaluation['episodes'] >= minimum_episodes and
                               evaluation['mean_passes'] >= threshold) else level
