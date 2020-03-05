import numpy as np
import gym
from dnc.dnc import DNC
import torch
import torch.optim as optim
import torch.nn as nn


gamma = 0.99 # discount factor for reward


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
      to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    elif isinstance(h, (list, )):
        return [repackage_hidden(v) for v in h]
    else:
        return tuple(repackage_hidden(v) for v in h)


def repackage_hidden_dnc(h):
    if h is None:
        return None

    (chx, mhxs, _) = h
    chx = repackage_hidden(chx)
    if type(mhxs) is list:
        mhxs = [dict([(k, repackage_hidden(v)) for k, v in mhx.items()]) for mhx in mhxs]
    else:
        mhxs = dict([(k, repackage_hidden(v)) for k, v in mhxs.items()])
    return (chx, mhxs, None)


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make('CartPole-v0')
input_size = 4
policy = DNC(4,200,gpu_id=0, output_size=1)
hidden = None
observation = env.reset()
episode_number = 0
reward_sum = 0
reset = False
done_reward_stack, x_stack, y_stack, done_stack = [], [], [], []
while episode_number <= 5000:
    env.render()
    x = torch.from_numpy(np.reshape(observation,[1,4])).unsqueeze(1)
    x = x.type(torch.FloatTensor)
    x = x.cuda()
    hidden = repackage_hidden_dnc(hidden)
    left_prob, hidden = policy(x, hidden, reset_experience=reset)
    reset = False
    action = 1 if np.random.uniform() < left_prob.item() else 0
    # record various intermediates (needed later for backprop)
    x_stack.append(x)
    y = 1 if action == 0 else 0
    y_stack.append(y)
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    done_stack.append(done * 1)
    done_reward_stack.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    optimizer = optim.Adam(policy.parameters(), lr=0.0001, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
    if done:
        episode_number += 1
        observation = env.reset()
        reset = True

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(x_stack)
        epy = np.vstack(y_stack)
        epr = np.vstack(done_reward_stack)
        epd = np.vstack(done_stack)
        x_stack, done_reward_stack, y_stack, done_stack = [], [], [], [] # reset array memory
        discounted_epr = discount_rewards(epr).astype('float32')
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)