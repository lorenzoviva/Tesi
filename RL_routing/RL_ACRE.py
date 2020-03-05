import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from models import *
import random
# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

plt.ion()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value', "entropy"])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 128)



        # actor's layer
        self.action_head = nn.Linear(128, 128) # 1

        # critic's layer
        self.value_head = nn.Linear(128, 32) # 1

        self.state_action_routing = nn.Sequential(
            Routing(d_cov=1, d_inp=4, d_out=4, n_out=8), #8
            Routing(d_cov=1, d_inp=4, d_out=2, n_inp=8, n_out=2),
        )

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_caps = self.action_head(x).view(x.shape[0],-1,1,4)

        # critic: evaluates being in the state s_t
        state_caps = self.value_head(x)

        for routing in self.state_action_routing:
            state_caps, action_caps, _ = routing(state_caps, action_caps)

        state_prob = F.softmax(state_caps, dim=-1)
        routed_state = state_caps.max(1)
        state_caps = routed_state[0]
        action_caps = action_caps[:,routed_state[1],:,:].squeeze()
        action_prob = F.softmax(action_caps, dim=-1)


        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_caps, state_prob


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value, state_prob = model(state.unsqueeze(0))

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value,  m.entropy()))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    act = []

    for (log_prob, value, entropy), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        if torch.isnan(entropy) or random.random() > 0.5:
            policy_losses.append(-log_prob * advantage)
        else:
            policy_losses.append(-log_prob / entropy)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        act.append(value)

    # reset gradients
    optimizer.zero_grad()



    # sum up all the values of policy_losses and value_losses

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.show()
    plt.pause(0.001)

def main():
    running_reward = 10

    # run inifinitely many episodes
    episode_durations = []
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):
            # env.render()
            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                episode_durations.append(t)
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            plot_durations(episode_durations)

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()