import numpy as np
import torch


def gae(rewards, values, next_values, terminated, done, gamma=.99, lam=.95):
    """Inputs [time, agent]. Bootstrap time limits but stop traces at any reset."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma * lam * (1 - done[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def clipped_loss(new_logp, old_logp, advantages, clip=.2):
    ratio = (new_logp - old_logp).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


def update(model, optimizer, batch, clip=.2, entropy_coef=.01, epochs=4, minibatch=128):
    obs, masks, actions, old_logp, advantages, returns = batch
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    results = []
    for _ in range(epochs):
        for indices in torch.randperm(len(obs), device=obs.device).split(minibatch):
            dist, value = model(obs[indices], masks[indices])
            logp = dist.log_prob(actions[indices])
            policy_loss = clipped_loss(logp, old_logp[indices], advantages[indices], clip)
            value_loss = .5 * (value - returns[indices]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + .5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(grad):
                raise FloatingPointError('Non-finite PPO gradient')
            optimizer.step()
            with torch.no_grad():
                lr = logp - old_logp[indices]
                kl = ((lr.exp() - 1) - lr).mean()
            results.append([float(policy_loss.detach()), float(value_loss.detach()),
                            float(entropy.detach()), float(kl), float(grad)])
    averages = np.mean(results, axis=0)
    with torch.no_grad():
        _, predictions = model(obs, masks)
        variance = returns.var(unbiased=False)
        explained = 1 - (returns - predictions).var(unbiased=False) / variance if variance > 1e-8 else None
    return dict(zip(('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'gradient_norm'), averages.tolist())) | {
        'explained_variance': float(explained) if explained is not None else None}
