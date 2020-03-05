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


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class SubNetwork(nn.Module):

    def __init__(self, shape):
        super(SubNetwork, self).__init__()
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x, new_params, action):
        with torch.no_grad():
            if action == 0:
                self.weight = nn.Parameter((self.weight * torch.randn(self.weight.shape)).clamp(-1,1))
            else:
                self.weight = nn.Parameter((self.weight / torch.randn(self.weight.shape)).clamp(-1,1))
            self.weight.clamp(-1,1)
                #self.weight = nn.Parameter((self.weight / new_params.repeat(128,1).transpose(1,0)))
            weights = self.weight.view(-1,x.shape[0])
        x = F.linear(x, weights)
        return x


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, actions):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 128)



        # actor's layer
        self.action_head = nn.Linear(128, 128) # 1

        # critic's layer
        self.value_head = nn.Linear(128, 32) # 1

        self.state_action_routing = nn.Sequential(
            Routing(d_cov=1, d_inp=4, d_out=4, n_out=8), #8
            Routing(d_cov=1, d_inp=4, d_out=4, n_inp=8, n_out=128),
        )

        self.sub_net = SubNetwork((2,2))
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, input):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(input))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_caps = self.action_head(x).view(x.shape[0],-1,1,4)

        # critic: evaluates being in the state s_t
        state_caps = self.value_head(x)

        for routing in self.state_action_routing:
            state_caps, action_caps, _ = routing(state_caps, action_caps)

        routed_state = state_caps.max(1)
        state_caps = routed_state[0]
        action_caps = action_caps[:,routed_state[1],:,:].squeeze()
        action_prob = F.softmax(action_caps, dim=-1)

        ms = Categorical(action_prob)

        # and sample an action using the distribution
        sub_action = ms.sample()
        final_action_probs = self.sub_net(input, action_caps, sub_action)
        output_probs = F.softmax(final_action_probs.squeeze(), dim=-1)
        #output_probs = F.softmax(self.sub_net(x, action_caps.squeeze().transpose(1,0)), dim=-1)
        # create a categorical distribution over the list of probabilities of actions
        # m = Categorical(action_prob)
        if output_probs[0] < 0 or output_probs[1] < 0:
            print("error")
        m = Categorical(output_probs)

        try:
            # and sample an action using the distribution
            action = m.sample()
        except RuntimeError:
            print("errro")

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_caps))

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action


model = Policy(2)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    action = model(state.unsqueeze(0))

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

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

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