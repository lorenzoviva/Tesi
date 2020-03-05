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
import torchvision.transforms as T
from PIL import Image
from heinsen_routing import *
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
    def __init__(self, w, h, common_nn_size=128, init_caps_size=4, init_caps_number=32, routing_layers=[(4,8)]):
        super(Policy, self).__init__()
        self.init_caps_size = init_caps_size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 4)

        self.affine1 = nn.Linear(4, common_nn_size)

        # actor's layer
        self.action_head = nn.Linear(common_nn_size, init_caps_number*init_caps_size) # 1

        # critic's layer
        self.value_head = nn.Linear(common_nn_size, init_caps_number) # 1

        self.state_action_routing = nn.Sequential()
        for i, (d_out, n_out) in enumerate(routing_layers):
            if i > 0:
                self.state_action_routing.add_module("routing" + str(i), Routing(d_cov=1, d_inp=routing_layers[i-1][0], d_out=d_out, n_out=n_out))
            else:
                self.state_action_routing.add_module("routing" + str(i), Routing(d_cov=1, d_inp=init_caps_size, d_out=d_out, n_out=n_out))

        if routing_layers:
            self.state_action_routing.add_module("routingfinal",Routing(d_cov=1, d_inp=routing_layers[-1][0], d_out=2, n_out=2))
        else:
            self.state_action_routing.add_module("routingfinal",Routing(d_cov=1, d_inp=init_caps_size, d_out=2, n_out=2))

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        x = F.relu(self.affine1(x))


        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_caps = self.action_head(x).view(x.shape[0],-1,1,self.init_caps_size)

        # critic: evaluates being in the state s_t
        state_caps = self.value_head(x)

        for routing in self.state_action_routing:
            state_caps, action_caps, _ = routing(state_caps, action_caps)

        routed_state = state_caps.max(1)
        state_caps = routed_state[0]
        action_caps = action_caps[:,routed_state[1],:,:].squeeze()
        action_prob = F.softmax(action_caps, dim=-1)


        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_caps

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)

    return resize(screen).unsqueeze(0)#.cuda()




def main():
    env.reset()
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    model = Policy(screen_width, screen_height)
    # optimizer = optim.Adam(model.parameters(), lr=3e-2)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = optim.RMSprop(model.parameters())

    eps = np.finfo(np.float32).eps.item()

    def select_action(state):
        # state = torch.from_numpy(state).float()
        probs, state_value = model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

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
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

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

    running_reward = 10

    # run inifinitely many episodes
    episode_durations = []
    for i_episode in count(1):

        # reset environment and episode reward
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        ep_reward = 0
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            observation, reward, done, _ = env.step(action)

            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                state = current_screen - last_screen

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


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


if __name__ == '__main__':
    main()