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
import matplotlib.animation as animation
# plt.rcParams['animation.convert_path'] = '/usr/bin/convert'

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

        # self.state_action_routing = nn.Sequential(
        #     Routing(d_cov=1, d_inp=4, d_out=4, n_out=4), #8
        #     Routing(d_cov=1, d_inp=4, d_out=2, n_inp=4, n_out=2),
        # )
        self.state_action_routing = nn.Sequential(
            Routing(d_cov=1, d_inp=4, d_out=2, n_out=1),
        )
        # self.state_action_routing = nn.Sequential(
        #     Routing(d_cov=1, d_inp=4, d_out=4, n_out=8),
        #     Routing(d_cov=1, d_inp=4, d_out=4, n_inp=8, n_out=4),
        #     Routing(d_cov=1, d_inp=4, d_out=2, n_inp=4, n_out=2),
        # )
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []



    def capsule_layer_to_image(self, state_caps, action_caps):
        PADDING = 4
        state_image = torch.cat([torch.zeros(state_caps.shape[0], PADDING), state_caps, torch.zeros(state_caps.shape[0], PADDING)], 1)
        action_image = torch.cat([torch.zeros(action_caps.shape[-1], PADDING), action_caps.permute(3,1,0,2).squeeze(-1).squeeze(-1), torch.zeros(action_caps.shape[-1], PADDING)], 1)
        image_padding = torch.zeros(PADDING,state_image.shape[-1])
        layer_image = torch.cat([image_padding, state_image, image_padding, action_image, image_padding]).permute(1, 0)
        # reshaped = F.interpolate(F.interpolate(layer_image.unsqueeze(0).repeat(3, 1, 1), scale_factor=10).permute(0, 2, 1), scale_factor=10).permute(0, 2, 1)
        # reshaped = F.interpolate(F.interpolate(layer_image.unsqueeze(0), scale_factor=10).permute(0, 2, 1), scale_factor=10).permute(0, 2, 1)
        reshaped = F.interpolate(F.interpolate(layer_image.unsqueeze(0), scale_factor=4).permute(0, 2, 1), scale_factor=4).squeeze(0)
        # reshaped = reshaped - reshaped.min()
        # reshaped = reshaped / reshaped.max()
        return reshaped

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
        action_prob = []
        layer_images = []
        state_capsules = []
        for routing in self.state_action_routing:
            routed_state = state_caps.max(-1)
            state_capsules.append(routed_state[0])
            action_cap = action_caps[:,routed_state[1],:,:].squeeze()
            layer_images.append(self.capsule_layer_to_image(state_caps,action_caps))
            action_prob.append(F.softmax(action_cap, dim=-1))
            state_caps, action_caps, _ = routing(state_caps, action_caps)

        layer_images.append(self.capsule_layer_to_image(state_caps,action_caps))
        routed_state = state_caps.max(-1)
        state_capsules.append(routed_state[0])
        action_cap = action_caps[:,routed_state[1],:,:].squeeze()
        action_prob.append(F.softmax(action_cap, dim=-1))


        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_capsules, layer_images


model = Policy()
# optimizer = optim.Adam(model.parameters(), lr=3e-2)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_values, images = model(state.unsqueeze(0))

    # create a categorical distribution over the list of probabilities of actions
    for i, (action_prob, state_value) in enumerate(zip(probs, state_values)):
        m = Categorical(action_prob)

        # and sample an action using the distribution
        action = m.sample()
        if len(model.saved_actions) <= i:
            model.saved_actions.append([])
        # save to action buffer
        if i + 1 == len(probs):
            model.saved_actions[i].append(SavedAction(m.log_prob(action), state_value))
        else:
            model.saved_actions[i].append(SavedAction(m.log_prob(action_prob), state_value))

    # the action to take (left or right)
    return action.item(), images


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
    for i, caps_reward in enumerate(model.rewards):
        R = 0
        for r in caps_reward[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            if len(returns) <= i:
                returns.append([])
            returns[i].insert(0, R)
    for i, expected in enumerate(returns):
        expected = torch.tensor(expected, dtype=torch.float32).cuda()
        expected = (expected - expected.mean()) / (expected.std() + eps)
        returns[i] = expected

    for expected, saved_action in zip(returns, saved_actions):
        for (log_prob, value), R in zip(saved_action, expected):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(torch.sum(-log_prob.cuda() * advantage))

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(torch.tensor(value.item(), dtype=torch.float32).cuda(), R))
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
    figure = plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    figure.show()

    plt.pause(0.001)

def plot_images(images, fig):
    max_image_height = -1
    im = []
    for image in images:
        if max_image_height < image.shape[1]:
            max_image_height = image.shape[1]
    if max_image_height != -1:
        for i, image in enumerate(images):
            # images[i] = torch.cat([torch.zeros(3, np.ceil((max_image_height - images[i].shape[1])/2).astype(np.int),image.shape[-1]),
            #                        images[i],
            #                        torch.zeros(3, np.floor((max_image_height - images[i].shape[1])/2).astype(np.int),image.shape[-1])], 1)

        # summary = torch.cat(images,-1)
            fig.add_subplot(1, len(images), i+1)
            # im = [plt.imshow(images[0].permute(1,2,0).detach().numpy(), interpolation='none', animated=True)]#, cmap='cividis')
            # fig.add_subplot(2, 1, 2)
            if len(image.shape) > 2:
                im += [plt.imshow(image.permute(1,2,0).detach().numpy(), interpolation='none', animated=True)]#, )
            else:
                im += [plt.imshow(image.permute(1,0).detach().numpy(), interpolation='none', animated=True, cmap='cividis')]
    # im = None
    #
    # plt.colorbar()
    # plt.show()
    return im

def main():
    running_reward = 10

    # run inifinitely many episodes
    episode_durations = []
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        gif_images = []
        fig = plt.figure()
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 1000):

            # select action from policy
            action, images = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)
            images.append(torch.from_numpy((env.render(mode='rgb_array')/255)).permute(2,0,1).float())
            im = plot_images(images, fig)
            gif_images.append(im)

            if args.render:
                env.render()
            if model.rewards:
                model.rewards[0].append(-state[2])
                model.rewards[1].append(reward)
            else:
                model.rewards = [[-state[2]],[reward]]

            ep_reward += reward
            if done:
                episode_durations.append(t)
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()
        plt.colorbar()
        ani = animation.ArtistAnimation(fig, gif_images, interval=300, blit=True,
                                        repeat_delay=500)
        ani.save('data/logs/episode' + str(i_episode) + ".gif", writer='imagemagick')
        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            plot_durations(episode_durations)

            # plt.show()



        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()