from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rondolab.environments.parallel import ACTION_SIZE, OBS_SIZE, action_mask, observation


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_SIZE, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh())
        self.actor = nn.Linear(96, ACTION_SIZE)
        self.critic = nn.Linear(96, 1)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, obs, mask):
        hidden = self.body(obs)
        logits = self.actor(hidden).masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def decisions(self, world, indices, scenario='adaptive', deterministic=True):
        obs = torch.tensor(np.stack([observation(world, i) for i in indices]), dtype=torch.float32,
                           device=next(self.parameters()).device)
        masks = torch.tensor(np.stack([action_mask(world, i, scenario) for i in indices]), device=obs.device)
        dist, values = self(obs, masks)
        acts = dist.probs.argmax(-1) if deterministic else dist.sample()
        result = {}
        for j, idx in enumerate(indices):
            p = world.players[idx]
            p.control = 'PPO'
            p.probabilities = dist.probs[j].cpu().tolist()
            p.value = float(values[j])
            result[idx] = int(acts[j])
        return result


def save_checkpoint(path, model, metadata, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema_version': 1, 'state_dict': model.cpu().state_dict(), 'metadata': metadata,
                'optimizer': optimizer.state_dict() if optimizer is not None else None}, path)


def load_checkpoint(path, device='cpu'):
    data = torch.load(path, map_location=device, weights_only=True)
    model = ActorCritic().to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, data['metadata']
