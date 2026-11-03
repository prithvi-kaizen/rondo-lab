from rondolab.simulation.config import Config

STAGES = {
    1: {'width': 15, 'height': 15, 'defender_speed': 1., 'defender_strategy': 'chase', 'max_touches': 0},
    2: {'width': 12, 'height': 12, 'defender_speed': 2.3, 'defender_strategy': 'cover_press', 'max_touches': 0},
    3: {'width': 12, 'height': 12, 'defender_speed': 2.3, 'max_touches': 2},
    4: {'width': 12, 'height': 12, 'defender_speed': 2.3, 'max_touches': 2},
    5: {'width': 12, 'height': 12, 'defender_speed': 2.3, 'max_touches': 2},
    6: {'defender_speed': 2.3, 'max_touches': 2},
}
NAMES = {1: 'Basic passing', 2: 'Under pressure', 3: 'Adaptive possession', 4: 'Learn defending',
         5: 'Adversarial populations', 6: 'Generalization'}


def level_config(level, base=None, rng=None):
    data = (base or Config()).model_dump() | STAGES[level] | {'curriculum': level}
    if level == 6 and rng is not None:
        size = float(rng.choice([10, 12, 15]))
        data.update(width=size, height=size, defender_strategy=str(rng.choice(['cover_press', 'lanes', 'nearest'])))
    return Config(**data)


def promotion(level, evaluation, threshold=4, minimum_episodes=10, team='possession'):
    """Promotion requires held-out football outcomes, never elapsed steps."""
    if evaluation['episodes'] < minimum_episodes:
        return level
    if team == 'pressing' or level == 4:
        qualified = evaluation.get('forced_turnover_rate', 0) >= .6 and (
            evaluation.get('mean_time_to_regain') is not None and evaluation['mean_time_to_regain'] <= 15)
    else:
        qualified = evaluation['mean_passes'] >= threshold and evaluation['mean_duration'] >= 8
    return min(level + 1, 6) if qualified else level
