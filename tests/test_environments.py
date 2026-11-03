import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test, parallel_seed_test

from rondolab.environments.parallel import PassingEnv, RondoParallelEnv, action_mask, observation
from rondolab.simulation.config import Config


def test_gym_contract():
    check_env(PassingEnv(), skip_render_check=True)


def test_parallel_contract():
    parallel_api_test(RondoParallelEnv(), num_cycles=80)
    parallel_seed_test(RondoParallelEnv, num_cycles=30)


def test_spaces_masks_rewards_and_truncation():
    env = RondoParallelEnv(Config(duration=5, defender_speed=0, max_hold=10))
    obs, _ = env.reset(seed=3)
    assert all(env.observation_space(a).contains(o) for a, o in obs.items())
    assert observation(env.world, 0).shape == (90,)
    assert action_mask(env.world, 0)[1] == 0
    for _ in range(25):
        obs, rewards, term, trunc, infos = env.step({a: 0 for a in env.agents})
    assert all(trunc.values()) and not any(term.values())
    assert not env.agents
    assert np.isclose(rewards['p0'], -rewards['p6'])
    assert sum(infos['p0']['reward_components'].values()) == rewards['p0']
