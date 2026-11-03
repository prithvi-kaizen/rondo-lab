import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rondolab.matchups import matchup, matrix
from rondolab.policies.neural import load_checkpoint
from rondolab.replay import Replay, generate_episode
from rondolab.simulation.config import PRESETS, Config

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / 'experiments'
app = FastAPI(title='Rondo Lab', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                   allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])
job_lock = threading.Lock()
job = None
metadata_cache = {}


def safe_checkpoint(identifier):
    root = EXPERIMENTS.resolve()
    path = (root / identifier).resolve()
    if not path.is_relative_to(root) or path.suffix != '.pt' or not path.is_file():
        raise HTTPException(404, 'Checkpoint not found')
    return path


def checkpoint_metadata(path):
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in metadata_cache:
        data = torch.load(path, map_location='cpu', weights_only=True)
        metadata_cache[key] = data.get('metadata')
    meta = metadata_cache[key]
    return {'environment_version': 1, 'generation': 0, 'compatibility': 'rondo-v1', **meta} if meta else None


@app.get('/api/policies')
def policies(include_drafts: bool = False):
    result = []
    for path in sorted(EXPERIMENTS.rglob('*.pt')) if EXPERIMENTS.exists() else []:
        if path.name == 'session.pt':
            continue
        meta = checkpoint_metadata(path)
        if meta and meta['environment_version'] == 2 and meta.get('config', {}).get('engine_revision') != '2.0' and not include_drafts:
            continue
        if meta:
            result.append({'id': str(path.relative_to(EXPERIMENTS)), **meta})
    return result


@app.get('/api/config')
def config():
    return {'defaults': Config().model_dump(), 'presets': PRESETS, 'schema': Config.model_json_schema()}


class EpisodeRequest(BaseModel):
    config: Config = Field(default_factory=Config)
    seed: int = Field(42, ge=0, le=2147483647)
    checkpoint: str | None = None
    opponent_checkpoint: str | None = None
    baseline: Literal['heuristic', 'simple', 'random'] = 'heuristic'


@app.post('/api/episodes')
def episode(request: EpisodeRequest):
    model, meta, opponent, opponent_meta = None, None, None, None
    c = request.config
    if request.checkpoint:
        model, meta = load_checkpoint(safe_checkpoint(request.checkpoint))
        if meta['environment_version'] == 1:
            c = Config(**(meta['config'] | {'environment_version': 1}))
        elif c.environment_version != 2:
            raise HTTPException(422, 'Select a v2 grid to replay a v2 checkpoint')
    if request.opponent_checkpoint:
        if not model:
            # A defender checkpoint can face scripted possession without a possession checkpoint.
            opponent, opponent_meta = load_checkpoint(safe_checkpoint(request.opponent_checkpoint))
            if opponent_meta['team'] != 'pressing':
                raise HTTPException(422, 'An unpaired opponent must be a pressing checkpoint')
            model, meta, opponent, opponent_meta = opponent, opponent_meta, None, None
        else:
            opponent, opponent_meta = load_checkpoint(safe_checkpoint(request.opponent_checkpoint))
            if opponent_meta['team'] == meta['team'] or opponent_meta['environment_version'] != meta['environment_version']:
                raise HTTPException(422, 'Choose opposing teams from the same environment version')
    if model and model.environment_version != c.environment_version:
        raise HTTPException(422, 'Checkpoint is incompatible with this football environment')
    return generate_episode(c, request.seed, model, meta, request.baseline, opponent,
                            opponent_metadata=opponent_meta)


@app.post('/api/replays/validate')
def validate_replay(replay: Replay):
    return replay


@app.get('/api/experiments')
def experiments():
    return [{'id': str(path.parent.relative_to(EXPERIMENTS)), **json.loads(path.read_text()),
             'population': json.loads((path.parent / 'population.json').read_text())
             if (path.parent / 'population.json').exists() else None}
            for path in sorted(EXPERIMENTS.rglob('metrics.json'))] if EXPERIMENTS.exists() else []


@app.get('/api/sessions')
def sessions():
    result = []
    for path in sorted(EXPERIMENTS.rglob('session.pt')) if EXPERIMENTS.exists() else []:
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['settings']['config'].get('engine_revision') != '2.0':
            continue
        result.append({'id': str(path.relative_to(EXPERIMENTS)), 'steps': data['steps'],
                       'generation': max(data['team_updates'].values()), **data['settings']})
    return result


class TrainingRequest(BaseModel):
    mode: Literal['possession', 'pressing', 'both'] = 'possession'
    scenario: Literal['basic', 'adaptive'] = 'basic'
    timesteps: int = Field(4096, ge=32, le=10000000)
    seed: int = Field(42, ge=0, le=2147483647)
    checkpoint_interval: int = Field(2048, ge=16, le=1000000)
    config: Config = Field(default_factory=Config)
    block_steps: int = Field(256, ge=16, le=8192)
    num_envs: int = Field(2, ge=1, le=16)
    resume: str | None = None
    device: Literal['cpu', 'auto', 'cuda', 'mps'] = 'cpu'
    curriculum_stage: int | None = Field(None, ge=1, le=6)
    auto_curriculum: bool = False
    evaluation_episodes: int = Field(2, ge=1, le=30)


@app.post('/api/training')
def start_training(request: TrainingRequest):
    global job
    if request.config.environment_version != 2:
        raise HTTPException(422, 'Legacy football supports playback; choose a v2 grid for new training')
    if request.mode == 'both' and request.scenario != 'adaptive' and not request.resume:
        raise HTTPException(422, 'Adversarial training requires adaptive six-player actions')
    resume_path = safe_checkpoint(request.resume) if request.resume else None
    if resume_path:
        data = torch.load(resume_path, map_location='cpu', weights_only=True)
        if data.get('kind') != 'training-session' or data.get('environment_version') != 2:
            raise HTTPException(422, 'Select a compatible v2 session.pt to resume optimizer and population state')
        if data['settings']['config'].get('engine_revision') != '2.0':
            raise HTTPException(422, 'Draft session predates finalized v2 football mechanics')
        if request.timesteps <= data['steps']:
            raise HTTPException(422, f'Resume target must exceed {data["steps"]} total steps')
    with job_lock:
        if job and job['process'].poll() is None:
            raise HTTPException(409, 'A training process is already running')
        name = f'v2-{request.mode}-{time.time_ns()}'
        output = EXPERIMENTS / name
        output.mkdir(parents=True)
        (output / 'config.json').write_text(request.config.model_dump_json())
        log = (output / 'process.log').open('w')
        command = [sys.executable, '-m', 'rondolab.train', '--mode', request.mode, '--scenario', request.scenario,
                   '--timesteps', str(request.timesteps), '--seed', str(request.seed), '--checkpoint-interval',
                   str(request.checkpoint_interval), '--output', str(output), '--config', str(output / 'config.json'),
                   '--block-steps', str(request.block_steps), '--num-envs', str(request.num_envs),
                   '--device', request.device, '--evaluation-episodes', str(request.evaluation_episodes)]
        if resume_path:
            command += ['--resume', str(resume_path)]
        if request.curriculum_stage:
            command += ['--curriculum-stage', str(request.curriculum_stage)]
        if request.auto_curriculum:
            command += ['--auto-curriculum']
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy() | {'RONDO_WORKER_LOCK': str(EXPERIMENTS / '.training-worker.lock')})
        log.close()
        job = {'id': name, 'process': process, 'output': output, 'requested': None}
    return {'id': name, 'status': 'running'}


@app.get('/api/training')
def training_status():
    with job_lock:
        if not job:
            return {'status': 'idle'}
        code = job['process'].poll()
        path = job['output'] / 'status.json'
        progress = json.loads(path.read_text()) if path.exists() else None
        if code is None:
            status = ('pausing' if job['requested'] == 'paused' else 'stopping') if job['requested'] else 'running'
        else:
            status = (progress or {}).get('state', 'completed') if code == 0 else 'failed'
        log = job['output'] / 'process.log'
        return {'id': job['id'], 'status': status, 'progress': progress if progress and progress.get('latest') else None,
                'detail': progress, 'resume': str((job['output'] / 'session.pt').relative_to(EXPERIMENTS))
                if (job['output'] / 'session.pt').exists() else None,
                'error': log.read_text()[-3000:] if status == 'failed' else None}


class ControlRequest(BaseModel):
    action: Literal['paused', 'cancelled']


@app.post('/api/training/control')
def training_control(request: ControlRequest):
    with job_lock:
        if job and job['process'].poll() is None:
            job['requested'] = request.action
            path = job['output'] / 'control.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'action': request.action}))
            temporary.replace(path)
    return {'status': 'requested', 'action': request.action}


@app.delete('/api/training')
def cancel_training():
    training_control(ControlRequest(action='cancelled'))
    # Compatibility with the original UI: wait briefly for a safe update boundary.
    if job and job['process'].poll() is None:
        try:
            job['process'].wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
    return {'status': 'cancelled' if job and job['process'].poll() is not None else 'stopping'}


class MatrixRequest(BaseModel):
    possession: list[str] = Field(default_factory=lambda: ['scripted'], min_length=1, max_length=4)
    defending: list[str] = Field(default_factory=lambda: ['scripted'], min_length=1, max_length=4)
    config: Config = Field(default_factory=Config)
    episodes: int = Field(3, ge=1, le=20)


def resolved(ids):
    return [str(safe_checkpoint(i)) if i != 'scripted' else 'scripted' for i in ids]


@app.post('/api/matchups')
def evaluate_matchups(request: MatrixRequest):
    if request.config.environment_version != 2:
        raise HTTPException(422, 'Matchup matrix requires an environment-v2 grid')
    try:
        result = matrix(resolved(request.possession), resolved(request.defending), request.config,
                        list(range(10001, 10001 + request.episodes)))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    for row in result['matchups']:
        for team in ('possession', 'defending'):
            if row[team] != 'scripted':
                row[team] = str(Path(row[team]).relative_to(EXPERIMENTS.resolve()))
    directory = EXPERIMENTS / 'evaluations'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'matrix-{time.time_ns()}.json').write_text(json.dumps(result, indent=2))
    return result


class FootballExperiment(MatrixRequest):
    experiment: Literal['geometry', 'coordination', 'adaptation', 'evolution', 'rewards']


@app.post('/api/football-experiments')
def football_experiment(request: FootballExperiment):
    heldout = list(range(10001, 10001 + request.episodes))
    if request.experiment in ('evolution', 'rewards'):
        return {'experiment': request.experiment, **evaluate_matchups(request)}
    rows = []
    if request.experiment == 'geometry':
        import numpy as np

        from rondolab.diagnostics import analyse_replay
        for label, c in [('Legacy 15m hexagon', Config(environment_version=1, width=15, height=15, ball_speed=10,
                                                      receiving_time=.24, loss_penalty=2)),
                         ('Easy 15m grid', Config(width=15, height=15)), ('Standard 12m grid', Config()),
                         ('Hard 10m grid', Config(width=10, height=10))]:
            metrics = [analyse_replay(generate_episode(c, seed)) for seed in heldout]
            keys = ['passes', 'duration', 'interceptions', 'average_pressure', 'average_nearest_defender_distance',
                    'average_pass_distance', 'adjacent_pass_fraction']
            rows.append({'label': label, 'config': c.model_dump(), 'episodes': len(heldout),
                         **{key: float(np.mean([m[key] for m in metrics])) for key in keys}})
    else:
        p_path = request.possession[0]
        pm, pmeta = load_checkpoint(safe_checkpoint(p_path)) if p_path != 'scripted' else (None, None)
        if pm and pm.environment_version != 2:
            raise HTTPException(422, 'Choose an environment-v2 possession checkpoint')
        variants = [('Chase', request.config.model_copy(update={'defender_strategy': 'chase'})),
                    ('Press and cover', request.config.model_copy(update={'defender_strategy': 'cover_press'}))]
        if request.experiment == 'adaptation':
            variants[0] = ('Weak chase', request.config.model_copy(update={'defender_strategy': 'chase', 'defender_speed': 1}))
        for label, c in variants:
            rows.append({'label': label, **matchup(pm, config=c, seeds=heldout, possession_metadata=pmeta)})
        for identifier in request.defending:
            if identifier == 'scripted':
                continue
            dm, dmeta = load_checkpoint(safe_checkpoint(identifier))
            if dm.environment_version != 2 or dmeta['team'] != 'pressing':
                raise HTTPException(422, 'Choose environment-v2 pressing checkpoints')
            rows.append({'label': identifier, **matchup(pm, dm, request.config, heldout,
                                                       possession_metadata=pmeta, defending_metadata=dmeta)})
    return {'experiment': request.experiment, 'seeds': heldout, 'results': rows,
            'note': 'Measured outcomes, not evidence of learned improvement. Geometry comparison shows each actual configuration.'}


if (ROOT / 'frontend/dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'frontend/dist', html=True), name='frontend')
