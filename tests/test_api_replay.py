import json
from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rondolab import api as api_module
from rondolab.api import app
from rondolab.replay import Replay, generate_episode
from rondolab.simulation.config import Config

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_api_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, 'EXPERIMENTS', tmp_path)
    monkeypatch.setattr(api_module, 'job', None)
    yield
    if api_module.job and api_module.job['process'].poll() is None:
        api_module.cancel_training()


def test_configuration_and_safe_paths():
    assert client.get('/api/config').json()['defaults']['width'] == 12
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


def test_geometry_experiment_reports_measured_duration():
    response = client.post('/api/football-experiments', json={'experiment': 'geometry', 'episodes': 1})
    assert response.status_code == 200
    rows = response.json()['results']
    assert len(rows) == 4
    assert all(0 < row['duration'] <= row['config']['duration'] + .05 for row in rows)


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


def test_matched_policy_pair_and_matrix(tmp_path):
    from rondolab.policies.neural import ActorCritic, save_checkpoint
    for team, name in [('possession', 'p.pt'), ('pressing', 'd.pt')]:
        save_checkpoint(tmp_path / name, ActorCritic(), {'team': team, 'scenario': 'adaptive',
                        'generation': 0, 'steps': 0, 'config': Config().model_dump()})
    response = client.post('/api/episodes', json={'checkpoint': 'p.pt', 'opponent_checkpoint': 'd.pt'})
    assert response.status_code == 200
    replay = response.json()
    assert replay['policy']['team'] == 'possession' and replay['opponent']['team'] == 'pressing'
    assert all(p['control'] == 'PPO' for p in replay['frames'][1]['players'])
    matrix = client.post('/api/matchups', json={'possession': ['p.pt'], 'defending': ['d.pt'], 'episodes': 1})
    assert matrix.status_code == 200
    assert matrix.json()['matchups'][0]['seeds'] == [10001]
    assert client.post('/api/training', json={'resume': 'p.pt'}).status_code == 422
    assert client.post('/api/episodes', json={'checkpoint': 'p.pt', 'opponent_checkpoint': 'p.pt'}).status_code == 422


def test_cooperative_pause_and_resume_from_api():
    import time
    body = {'mode': 'both', 'scenario': 'adaptive', 'timesteps': 128, 'block_steps': 16,
            'checkpoint_interval': 16, 'num_envs': 1, 'evaluation_episodes': 1}
    assert client.post('/api/training', json=body).status_code == 200
    assert client.post('/api/training/control', json={'action': 'paused'}).status_code == 200
    def completed_status():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = client.get('/api/training').json()
            if status['status'] not in ('running', 'pausing', 'stopping'):
                return status
            time.sleep(.05)
        raise AssertionError('Worker did not reach a safe checkpoint boundary')
    paused = completed_status()
    assert paused['status'] == 'paused', paused
    saved = client.get('/api/sessions').json()
    assert saved and saved[0]['steps'] == paused['progress']['steps']
    target = saved[0]['steps'] + 16
    assert client.post('/api/training', json=body | {'resume': paused['resume'], 'timesteps': target}).status_code == 200
    resumed = completed_status()
    assert resumed['status'] == 'completed', resumed
    assert resumed['progress']['steps'] == target
