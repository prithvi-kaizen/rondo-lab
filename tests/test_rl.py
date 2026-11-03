import numpy as np
import torch

from rondolab.evaluate import evaluate
from rondolab.policies.neural import ActorCritic, load_checkpoint, save_checkpoint
from rondolab.rl.ppo import clipped_loss, gae
from rondolab.train import train


def test_gae_terminal_vs_time_limit():
    r = np.array([[1.], [2.]], dtype=np.float32)
    v = np.zeros_like(r)
    n = np.array([[0.], [10.]])
    term = np.array([[0.], [1.]])
    done = np.array([[0.], [1.]])
    a, _returns = gae(r, v, n, term, done, gamma=1, lam=1)
    assert np.allclose(a[:, 0], [3, 2])
    a, _ = gae(r, v, n, term * 0, done, gamma=1, lam=1)
    assert np.allclose(a[:, 0], [13, 12])


def test_clipped_loss_and_masked_gradients(tmp_path):
    assert np.isclose(float(clipped_loss(torch.tensor([np.log(2)]), torch.tensor([0.]),
                                        torch.tensor([1.]))), -1.2)
    model = ActorCritic()
    obs = torch.zeros((4, 110))
    masks = torch.zeros((4, 12), dtype=torch.bool)
    masks[:, :2] = True
    dist, values = model(obs, masks)
    assert dist.probs.shape == (4, 12) and values.shape == (4,)
    assert (dist.probs[:, 2:] == 0).all()
    actions = dist.sample()
    assert (actions < 2).all()
    loss = -dist.log_prob(actions).mean() + values.pow(2).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, {'steps': 0})
    loaded, meta = load_checkpoint(path)
    assert meta['steps'] == 0
    assert torch.equal(loaded(obs, masks)[0].probs, dist.probs)


def test_real_training_smoke_and_evaluation_no_mutation(tmp_path):
    history = train(timesteps=32, output=tmp_path, checkpoint_interval=16)
    assert history and history[0]['gradient_norm'] > 0
    model, _meta = load_checkpoint(tmp_path / 'possession-000032.pt')
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = evaluate(model=model, seeds=[10001], config=None)
    assert result['episodes'] == 1
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    initial, _ = load_checkpoint(tmp_path / 'possession-000000.pt')
    assert any(not torch.equal(v, initial.state_dict()[k]) for k, v in model.state_dict().items())
