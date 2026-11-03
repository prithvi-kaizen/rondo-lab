import json
from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rondolab.api import app
from rondolab.replay import Replay, generate_episode
from rondolab.simulation.config import Config

client = TestClient(app)


def test_configuration_and_safe_paths():
    assert client.get('/api/config').json()['defaults']['width'] == 15
    assert client.post('/api/episodes', json={'config': {'width': -1}}).status_code == 422
    assert client.post('/api/episodes', json={'checkpoint': '../../outside.pt'}).status_code == 404
    with pytest.raises(ValidationError):
        Config(defender_strategy='future_knowledge')


def test_replay_roundtrip_and_physical_displacement():
    replay = generate_episode(Config(duration=10), seed=42)
    assert Replay.model_validate_json(json.dumps(replay)).seed == 42
    assert client.post('/api/replays/validate', json=replay).status_code == 200
    assert replay['metrics']['passes'] > 0
    frames = replay['frames']
    for a, b in pairwise(frames):
        displacement = np.linalg.norm(np.array(b['ball']['position']) - a['ball']['position'])
        assert displacement <= 10 * .05 + .001
    bad = dict(replay, schema_version=99)
    assert client.post('/api/replays/validate', json=bad).status_code == 422


def test_training_job_lifecycle():
    response = client.post('/api/training', json={'timesteps': 10000})
    assert response.status_code == 200
    assert client.get('/api/training').json()['status'] == 'running'
    assert client.post('/api/training', json={'timesteps': 32}).status_code == 409
    assert client.delete('/api/training').status_code == 200
    assert client.get('/api/training').json()['status'] == 'cancelled'


def test_supported_training_modes(tmp_path):
    from rondolab.train import train
    for mode, scenario in [('possession', 'adaptive'), ('pressing', 'basic'), ('both', 'adaptive')]:
        history = train(mode=mode, scenario=scenario, timesteps=32, output=tmp_path / mode)
        assert history[-1]['steps'] == 32
        assert (tmp_path / mode / f'{history[-1]["team"]}-000032.pt').exists()
