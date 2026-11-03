import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rondolab.replay import Replay, generate_episode
from rondolab.simulation.config import PRESETS, Config

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / 'experiments'
app = FastAPI(title='Rondo Lab', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                   allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])
job_lock = threading.Lock()
job = None


def safe_checkpoint(identifier):
    root = EXPERIMENTS.resolve()
    path = (root / identifier).resolve()
    if not path.is_relative_to(root) or path.suffix != '.pt' or not path.is_file():
        raise HTTPException(404, 'Checkpoint not found')
    return path


def policies():
    result = []
    if EXPERIMENTS.exists():
        for path in sorted(EXPERIMENTS.rglob('*.pt')):
            from rondolab.policies.neural import load_checkpoint

            _, meta = load_checkpoint(path)
            result.append({'id': str(path.relative_to(EXPERIMENTS)), **meta})
    return result


@app.get('/api/config')
def config():
    return {'defaults': Config().model_dump(), 'presets': PRESETS, 'schema': Config.model_json_schema()}


@app.get('/api/policies')
def list_policies():
    return policies()


class EpisodeRequest(BaseModel):
    config: Config = Field(default_factory=Config)
    seed: int = Field(42, ge=0, le=2147483647)
    checkpoint: str | None = None
    baseline: str = 'heuristic'


@app.post('/api/episodes')
def episode(request: EpisodeRequest):
    if request.baseline not in ('heuristic', 'simple', 'random'):
        raise HTTPException(422, 'Invalid baseline')
    model, meta = None, None
    if request.checkpoint:
        from rondolab.policies.neural import load_checkpoint

        model, meta = load_checkpoint(safe_checkpoint(request.checkpoint))
    return generate_episode(request.config, request.seed, model, meta, request.baseline)


@app.post('/api/replays/validate')
def validate_replay(replay: Replay):
    return replay


@app.get('/api/experiments')
def experiments():
    return [{'id': str(path.parent.relative_to(EXPERIMENTS)), **json.loads(path.read_text())}
            for path in sorted(EXPERIMENTS.rglob('metrics.json'))] if EXPERIMENTS.exists() else []


class TrainingRequest(BaseModel):
    mode: str = 'possession'
    scenario: str = 'basic'
    timesteps: int = Field(4096, ge=32, le=10000000)
    seed: int = Field(42, ge=0, le=2147483647)
    checkpoint_interval: int = Field(2048, ge=32, le=1000000)
    config: Config = Field(default_factory=Config)


@app.post('/api/training')
def start_training(request: TrainingRequest):
    global job
    if request.mode not in ('possession', 'pressing', 'both') or request.scenario not in ('basic', 'adaptive'):
        raise HTTPException(422, 'Unsupported training mode or scenario')
    with job_lock:
        if job and job['process'].poll() is None:
            raise HTTPException(409, 'A training process is already running')
        name = f'{request.mode}-{time.time_ns()}'
        output = EXPERIMENTS / name
        output.mkdir(parents=True)
        (output / 'config.json').write_text(request.config.model_dump_json())
        log = (output / 'process.log').open('w')
        command = [sys.executable, '-m', 'rondolab.train', '--mode', request.mode,
                   '--scenario', request.scenario, '--timesteps', str(request.timesteps),
                   '--seed', str(request.seed), '--checkpoint-interval', str(request.checkpoint_interval),
                   '--output', str(output), '--config', str(output / 'config.json')]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy())
        log.close()
        job = {'id': name, 'process': process, 'output': output, 'cancelled': False}
    return {'id': name, 'status': 'running'}


@app.get('/api/training')
def training_status():
    with job_lock:
        if not job:
            return {'status': 'idle'}
        code = job['process'].poll()
        status = 'running' if code is None else ('cancelled' if job['cancelled'] else
                                                 'completed' if code == 0 else 'failed')
        path = job['output'] / 'status.json'
        log = job['output'] / 'process.log'
        return {'id': job['id'], 'status': status, 'progress': json.loads(path.read_text()) if path.exists() else None,
                'error': log.read_text()[-3000:] if status == 'failed' else None}


@app.delete('/api/training')
def cancel_training():
    with job_lock:
        if job and job['process'].poll() is None:
            job['cancelled'] = True
            job['process'].terminate()
            job['process'].wait(timeout=10)
    return {'status': 'cancelled'}


if (ROOT / 'frontend/dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'frontend/dist', html=True), name='frontend')
