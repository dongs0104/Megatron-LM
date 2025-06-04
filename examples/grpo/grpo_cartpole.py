import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, x, actions):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


def collect_trajectories(env, policy, steps, device):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    trajectories = []
    rewards = []
    log_probs = []
    actions = []
    dones = []
    states = []
    episode_rewards = []

    for _ in range(steps):
        with torch.no_grad():
            action, log_prob = policy.act(obs)
        next_obs, reward, done, _ = env.step(action.cpu().item())
        states.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        dones.append(done)
        episode_rewards.append(reward)

        obs = torch.tensor(next_obs if not done else env.reset(), dtype=torch.float32, device=device)
        if done:
            trajectories.append(episode_rewards)
            episode_rewards = []

    if episode_rewards:
        trajectories.append(episode_rewards)

    batch = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "rewards": torch.tensor(rewards, device=device),
        "dones": torch.tensor(dones, device=device, dtype=torch.bool),
    }
    return batch, trajectories


def compute_returns(batch, gamma):
    returns = []
    R = 0.0
    for reward, done in zip(reversed(batch["rewards"]), reversed(batch["dones"])):
        if done:
            R = 0.0
        R = reward + gamma * R
        returns.insert(0, R)
    return torch.stack(returns)


def compute_advantages(returns, values):
    return returns - values


def group_relative_advantages(advantages, groups):
    grouped = {}
    for idx, g in enumerate(groups):
        grouped.setdefault(g, []).append(idx)

    rel_adv = torch.zeros_like(advantages)
    for g, idxs in grouped.items():
        group_adv = advantages[idxs]
        mean_adv = group_adv.mean()
        rel_adv[idxs] = group_adv - mean_adv
    return rel_adv


def grpo_update(policy, optimizer, batch, returns, advantages, eps_clip, groups):
    states = batch["states"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]

    rel_adv = group_relative_advantages(advantages, groups)

    log_probs, entropy = policy.evaluate_actions(states, actions)
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * rel_adv
    surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * rel_adv
    loss = -(torch.min(surr1, surr2)).mean() - 0.01 * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO CartPole Example")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    env = gym.make(args.env)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(obs_size, action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        batch, trajectories = collect_trajectories(env, policy, args.steps, device)
        with torch.no_grad():
            values = torch.zeros(len(batch["states"]), device=device)
        returns = compute_returns(batch, args.gamma)
        advantages = compute_advantages(returns, values)
        groups = []
        t = 0
        for ep_idx, ep in enumerate(trajectories):
            for _ in ep:
                groups.append(ep_idx)
                t += 1
        grpo_update(policy, optimizer, batch, returns, advantages, args.eps_clip, groups)
        avg_reward = np.mean([sum(ep) for ep in trajectories])
        print(f"Epoch {epoch+1}: average reward {avg_reward:.2f}")
    env.close()
