# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) Tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__.
**Task**
The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. You can find an
official leaderboard with various algorithms and visualizations at the
`Gym website <https://gym.openai.com/envs/CartPole-v0>`__.
.. figure:: /_static/img/cartpole.gif
   :alt: cartpole
   cartpole
As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, rewards are +1 for every incremental timestep and the environment
terminates if the pole falls over too far or the cart moves more then 2.4
units away from center. This means better performing scenarios will run
for longer duration, accumulating larger return.
The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
However, neural networks can solve the task purely by looking at the
scene, so we'll use a patch of the screen centered on the cart as an
input. Because of this, our results aren't directly comparable to the
ones from the official leaderboard - our task is much harder.
Unfortunately this does slow down the training, because we have to
render all the frames.
Strictly speaking, we will present the state as the difference between
the current screen patch and the previous one. This will allow the agent
to take the velocity of the pole into account from one image.
**Packages**
First, let's import needed packages. Firstly, we need
`gym <https://gym.openai.com/docs>`__ for the environment
(Install using `pip install gym`).
We'll also use the following from PyTorch:
-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)
-  utilities for vision tasks (``torchvision`` - `a separate
   package <https://github.com/pytorch/vision>`__).
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from models import *
from pytorch_extras import RAdam
env = gym.make('CartPole-v0').unwrapped
from torch.autograd import Variable

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'time'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.
#
class RoutingModel(nn.Module):
    """
    Args:
        n_objs: int, number of objects to detect.
        n_parts: int, number of parts to detect.
        d_chns: int, number of channels in initial convolutions.

    Input:
        images: [..., 2, m, n] stacked smallNORB L and R m x n images.

    Output:
        a_out: [..., n_objs] object scores.
        mu_out: [..., n_objs, 4, 4] object poses.
        sig2_out: [..., n_objs, 4, 4] object pose variances.
    """
    def __init__(self, n_objs, n_parts, d_chns, n_out):
        super().__init__()
        self.convolve = nn.Sequential(
            *[m for (inp_ch, out_ch, stride) in zip([5] + [d_chns] * 5,  [d_chns] * 6,  [1, 2] * 3)
              for m in [nn.BatchNorm2d(inp_ch), nn.Conv2d(inp_ch, out_ch, 3, stride), Swish()]]
        )
        self.compute_a = nn.Sequential(nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, n_parts, 1))
        self.compute_mu = nn.Sequential(nn.BatchNorm2d(d_chns), nn.Conv2d(d_chns, n_parts * 4 * 4, 1))
        self.routings = nn.Sequential(
            Routing(d_cov=4, d_inp=4, d_out=4, n_out=n_parts),
            Routing(d_cov=4, d_inp=4, d_out=4, n_inp=n_parts, n_out=n_objs),
        )
        #self.head = nn.Linear(n_objs, n_out)
        self.head = nn.Linear(d_chns<<2, n_out)

        for conv in [m for m in self.convolve if type(m) == nn.Conv2d]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        self.n_out = n_out

    def add_coord_grid(self, x):
        h, w = x.shape[-2:]
        coord_grid = torch.stack((
            torch.linspace(-1.0, 1.0, steps=h, device=x.device)[:, None].expand(-1, w),
            torch.linspace(-1.0, 1.0, steps=w, device=x.device)[None, :].expand(h, -1),
        )).expand([*x.shape[:-3], -1, -1, -1])
        return torch.cat((x, coord_grid), dim=-3)

    def forward(self, images):
        x = self.add_coord_grid(images).cuda(device)           # [bs, (2 + 2), m, n]
        x = self.convolve(x)                                   # [bs, d_chns, m', n']

        a = self.compute_a(x)                                  # [bs, n_parts, m', n']
        a = a.view(a.shape[0], -1)                             # [bs, (n_parts * m' * n')]

        mu = self.compute_mu(x)                                # [bs, (n_parts * 4 * 4), m', n']
        mu = mu.view([mu.shape[0], -1, 4, 4, *mu.shape[-2:]])  # [bs, n_parts, 4, 4, m', n']
        mu = mu.permute(0, 1, 4, 5, 2, 3).contiguous()         # [bs, n_parts, m', n', 4, 4]
        mu = mu.view(mu.shape[0], -1, 4, 4)                    # [bs, (n_parts * m' * n'), 4, 4]

        for routing in self.routings:
            a, mu, sig2 = routing(a, mu)
        a_max = a.max(1)[1]
        mu_a_max = mu.gather(1,a_max.repeat(np.prod(mu.shape[2:])).view((mu.shape[0], 1) +mu.shape[2:] ))
        # a = self.head(a)
        a = self.head(mu_a_max.view(mu_a_max.shape[0], np.prod(mu_a_max.shape[1:])))
        return a, mu, sig2

class Generator(nn.Module):
    def __init__(self, ngpu, input_size, feature_map_size, channel_size):
        super(Generator, self).__init__()
        nz = input_size
        ngf = feature_map_size
        nc = channel_size
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Decoder(nn.Module):
    """
    Implement Decoder structure in section 4.1, Figure 2 to reconstruct a digit
    from the `DigitCaps` layer representation.
    The decoder network consists of 3 fully connected layers. For each
    [10, 16] output, we mask out the incorrect predictions, and send
    the [16,] vector to the decoder network to reconstruct a [784,] size
    image.
    This Decoder network is used in training and prediction (testing).
    """

    def __init__(self, num_classes, output_unit_size, input_width,
                 input_height, num_conv_in_channel, cuda_enabled):
        """
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 (or 3072 for CIFAR10) neurons each.
        """
        super(Decoder, self).__init__()

        self.cuda_enabled = cuda_enabled

        fc1_output_size = 512
        fc2_output_size = 1024
        self.fc3_output_size = input_width * input_height * num_conv_in_channel
        self.fc1 = nn.Linear(num_classes * output_unit_size, fc1_output_size) # input dim 10 * 16.
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
        self.fc3 = nn.Linear(fc2_output_size, self.fc3_output_size)
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target):
        """
        We send the outputs of the `DigitCaps` layer, which is a
        [batch_size, 10, 16] size tensor into the Decoder network, and
        reconstruct a [batch_size, fc3_output_size] size tensor representing the image.
        Args:
            x: [batch_size, 10, 16] The output of the digit capsule.
            target: [batch_size, 10] One-hot MNIST dataset labels.
        Returns:
            reconstruction: [batch_size, fc3_output_size] Tensor of reconstructed images.
        """
        batch_size = target.size(0)

        """
        First, do masking.
        """
        # Method 1: mask with y.
        # Note: we have not implement method 2 which is masking with true label.
        # masked_caps shape: [batch_size, 10, 16, 1]
        masked_caps = self.mask(x, self.cuda_enabled)

        """
        Second, reconstruct the images with 3 Fully Connected layers.
        """
        # vector_j shape: [batch_size, 160=10*16]
        vector_j = masked_caps.view(x.size(0), -1) # reshape the masked_caps tensor

        # Forward pass of the network
        fc1_out = self.relu(self.fc1(vector_j))
        fc2_out = self.relu(self.fc2(fc1_out)) # shape: [batch_size, 1024]
        reconstruction = self.sigmoid(self.fc3(fc2_out)) # shape: [batch_size, fc3_output_size]

        assert reconstruction.size() == torch.Size([batch_size, self.fc3_output_size])

        return reconstruction

    @staticmethod
    def mask(out_digit_caps, cuda_enabled=True):
        """
        In the paper, they mask out all but the activity vector of the correct digit capsule.
        This means:
        a) during training, mask all but the capsule (1x16 vector) which match the ground-truth.
        b) during testing, mask all but the longest capsule (1x16 vector).
        Args:
            out_digit_caps: [batch_size, 10, 16] Tensor output of `DigitCaps` layer.
        Returns:
            masked: [batch_size, 10, 16, 1] The masked capsules tensors.
        """
        # a) Get capsule outputs lengths, ||v_c||
        v_length = torch.sqrt((out_digit_caps**2).sum(dim=2))

        # b) Pick out the index of longest capsule output, v_length by
        # masking the tensor by the max value in dim=1.
        _, max_index = v_length.max(dim=1)
        max_index = max_index.data

        # Method 1: masking with y.
        # c) In all batches, get the most active capsule
        # It's not easy to understand the indexing process with max_index
        # as we are 3D animal.
        batch_size = out_digit_caps.size(0)
        masked_v = [None] * batch_size # Python list
        for batch_ix in range(batch_size):
            # Batch sample
            sample = out_digit_caps[batch_ix]

            # Masks out the other capsules in this sample.
            v = Variable(torch.zeros(sample.size()))
            if cuda_enabled:
                v = v.cuda()

            # Get the maximum capsule index from this batch sample.
            max_caps_index = max_index[batch_ix]
            v[max_caps_index] = sample[max_caps_index]
            masked_v[batch_ix] = v # append v to masked_v

        # Concatenates sequence of masked capsules tensors along the batch dimension.
        masked = torch.stack(masked_v, dim=0)

        return masked

######################################################################
# Input extraction
# ^^^^^^^^^^^^^^^^
#
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
#

resize = T.Compose([T.ToPILImage(),
                    T.Resize(64, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.265)
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
    return resize(screen).unsqueeze(0).to(device)

# def get_screen():
#     # Returned screen requested by gym is 400x600x3, but is sometimes larger
#     # such as 800x1200x3. Transpose it into torch order (CHW).
#     screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#     # Cart is in the lower half, so strip off the top and bottom of the screen
#     _, screen_height, screen_width = screen.shape
#     screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
#     view_width = int(screen_width * 0.6)
#     cart_location = get_cart_location(screen_width)
#     if cart_location < view_width // 2:
#         slice_range = slice(view_width)
#     elif cart_location > (screen_width - view_width // 2):
#         slice_range = slice(-view_width, None)
#     else:
#         slice_range = slice(cart_location - view_width // 2,
#                             cart_location + view_width // 2)
#     # Strip off the edges, so that we have a square image centered on a cart
#     screen = screen[:, :, slice_range]
#     # Convert to float, rescale, convert to torch tensor
#     # (this doesn't require a copy)
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     # Resize, and add a batch dimension (BCHW)
#     return resize(screen).unsqueeze(0).to(device)
#

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()


######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = RoutingModel(2, 15, 4, n_actions).cuda(device) # RoutingLayer(screen_height, screen_width, n_actions).to(device)
target_net = RoutingModel(2, 15, 4, n_actions).cuda(device) # RoutingLayer(screen_height, screen_width, n_actions).to(device)
# decoder = Decoder(2, 16, 40, 90, 1, True)
generator = Generator(1, 16, 8, 3).cuda(device)


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = RAdam(policy_net.parameters(), lr=5e-4)
optimizer = optim.RMSprop(policy_net.parameters(), lr=5e-4)  #,lr=5e-4
# decoder_optimizer = RAdam(decoder.parameters(), lr=5e-4)


# optimizer_g = RAdam(generator.parameters(), lr=5e-4)
optimizer_g = optim.Adam(generator.parameters(), lr=0.002, betas=( 0.5, 0.999))
# optimizer_g = optim.Adam(generator.parameters(), lr=2e-6, betas=( 0.5, 0.999))
# optimizer_g = RAdam(generator.parameters(), lr=5e-4)
#
memory = ReplayMemory(10000)


steps_done = 0



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state)[0].max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
episode_rewards = []
episode_reward = []

def plot_durations(generated_pre_transition, generated_post_transition, batch):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.plot(np.array(episode_rewards))
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(99)*means[0], means))
        plt.plot(means.numpy())
    if generated_post_transition is not None:
        randx = random.randint(0, generated_post_transition.shape[0] - 1)
        randi = randx
        for i in range(randx):
            if batch.next_state[i] is None:
                randi += 1
        fig = plt.figure(3)
        plt.clf()
        original_post_images = torch.cat([s for s in batch.next_state if s is not None])
        original_pre_images = torch.cat(batch.state)
        target_out = policy_net(original_post_images)
        post_action_batch = target_out[0].max(1)[1].repeat(1,16).reshape(target_out[0].shape[0],1,4,4)
        policy_out = policy_net(original_pre_images)
        pre_action_batch = torch.cat(batch.action).repeat(1,16).reshape(policy_out[0].shape[0],1,4,4)
        target_out_deformed = F.interpolate(F.interpolate(target_out[1].gather(1, post_action_batch)[randx].repeat(3,1,1).cpu(), scale_factor=16).permute(0,2,1), scale_factor=16).permute(0,2,1)
        policy_out_deformed = F.interpolate(F.interpolate(policy_out[1].gather(1, pre_action_batch)[randi].repeat(3,1,1).cpu(), scale_factor=16).permute(0,2,1), scale_factor=16).permute(0,2,1)
        image_generated_post_transition = generated_post_transition[randx].cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        image_generated_pre_trainsition = generated_pre_transition[randi].cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        image_original_post_transition = original_post_images[randx].cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        # image_generated_and_original = (image_original_post_transition + image_generated_post_transition) / 2
        image_out_latent_vector = target_out_deformed.permute(1, 2, 0).detach().numpy()
        image_in_latent_vector = policy_out_deformed.permute(1, 2, 0).detach().numpy()
        image_original_pre_transition = original_pre_images[randi].cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        fig.add_subplot(2, 3, 1)
        plt.title('Original')
        plt.imshow(image_original_post_transition, interpolation='none')
        fig.add_subplot(2, 3, 2)
        plt.title('reconstructed post transaction')#+"\n from vector:\n" + str(target_out_deformed))
        plt.imshow(image_generated_post_transition, interpolation='none')
        fig.add_subplot(2, 3, 5)
        plt.title('reconstructed pre transaction')#+"\n from vector:\n" + str(target_out_deformed))
        plt.imshow(image_generated_pre_trainsition, interpolation='none')
        fig.add_subplot(2, 3, 4)
        plt.title('Original')
        plt.imshow(image_original_pre_transition, interpolation='none')
        fig.add_subplot(2, 3, 3)
        plt.title(str(randx) + (" --> " if target_out[0].max(1)[1][randx] else " <--") + str(batch.time[randi]))#+"\n from vector:\n" + str(target_out_deformed))
        plt.imshow(image_out_latent_vector, interpolation='none')
        fig.add_subplot(2, 3, 6)
        plt.title(str(randi) + (" --> " if batch.action[randi] else " <--") + str(batch.time[randi]))#+"\n from vector:\n" + str(target_out_deformed))
        plt.imshow(image_in_latent_vector, interpolation='none')


    # vstack1 = np.vstack([image_generated_post_transition, image_generated_and_original])
        # vstack2 = np.vstack([image_out_latent_vector, image_original_post_transition])
        # hstack = np.hstack([vstack1,vstack2])
        # plt.imshow(hstack,interpolation='none')
        plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated


    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# def plot_durations(generated_image, batch):
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     plt.plot(np.array(episode_rewards))
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.ones(99)*means[0], means))
#         plt.plot(means.numpy())
#     if generated_image is not None:
#         randi = random.randint(0,generated_image.shape[0]-1)
#         plt.figure(3)
#         plt.clf()
#         original_images = torch.cat(batch.state)
#         policy_out = policy_net(original_images)
#         action_batch = torch.cat(batch.action).repeat(1,16).reshape(32,1,4,4)
#         policy_net_deformed = F.interpolate(F.interpolate(policy_out[1].gather(1, action_batch)[randi].repeat(3,1,1).cpu(), scale_factor=16).permute(0,2,1), scale_factor=16).permute(0,2,1)
#         vstack1 = np.vstack([generated_image[randi].cpu().squeeze(0).permute(1, 2, 0).detach().numpy(), (original_images[randi].cpu().squeeze(0).permute(1, 2, 0).detach().numpy()+generated_image[randi].cpu().squeeze(0).permute(1, 2, 0).detach().numpy())/2])
#         vstack2 = np.vstack([policy_net_deformed.permute(1, 2, 0).detach().numpy(), original_images[randi].cpu().squeeze(0).permute(1, 2, 0).detach().numpy()])
#         hstack = np.hstack([vstack1,vstack2])
#         plt.imshow(hstack,interpolation='none')
#         plt.title('Example reconstructed screen: ' + str(randi) + (" -->" if batch.action[randi] else " <--")  )#+"\n from vector:\n" + str(policy_net_deformed))
#         plt.show()
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
#
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())
######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                             batch.next_state)), device=device, dtype=torch.uint8)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                        if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#     action_batch = action_batch.repeat(1,16).reshape((BATCH_SIZE,1,4,4))
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     policy_out = policy_net(state_batch)
#
#     state_action_values = policy_out[1].gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros((BATCH_SIZE,1,4,4), device=device)
#     target_out = target_net(non_final_next_states)
#     target_max = target_out[0].max(1)[1].repeat(1,1).reshape((target_out[0].shape[0],1)).repeat(1,16).reshape((target_out[0].shape[0], 1, 4, 4))
#
#     #target_actions= torch.zeros(target_max.shape+ (2,), device=device)
#     #target_actions.gather(1, target_max)
#     #target_actions = torch.zeros((BATCH_SIZE,2), device=device)
#     #target_actions.index_select(0,target_max)
#     next_state_values[non_final_mask] = target_out[1].gather(1, target_max)#target_out[1].gather(target_out[0].max(1)[0].detach())
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch.repeat(1,1).repeat(16,1).reshape((BATCH_SIZE,1,4,4))
#
#     print(str(optimizer.state_dict().keys()))
#
#
#     # for key in optimizer.state_dict()["state"].keys():
#         # print(str(key) + ":" + str(optimizer.state_dict()["state"][key].keys() )+ " " + str(optimizer.state_dict()["state"][key]["square_avg"].size()))
#     # print(str(optimizer.state_dict()["param_groups"][0]))
#
#     # Compute Huber loss
#     target_actions = torch.ones(state_action_values.shape, device=device)
#     #loss = F.smooth_l1_loss(state_action_values, target_actions)
#     # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
#
#
#     losses = -expected_state_action_values.unsqueeze(1) * F.log_softmax(state_action_values, dim=-1)  # CE
#     loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch
#
# # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer.step()
######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    transitions_batch = []
    for next_state in batch.next_state:
        if next_state is not None:
            transitions_batch.append(next_state)
    reward_batch = torch.cat(batch.reward)

    state_batch = torch.cat(batch.state)
    policy_out = policy_net(state_batch)

    target_out = target_net(non_final_next_states)

    # generator optimization

    action_batch = torch.cat(batch.action).repeat(1,16).reshape(BATCH_SIZE,1,4,4)
    next_state_vector = torch.zeros((BATCH_SIZE, 1 ,4,4), device=device)

    pre_transaction_input_feature_map = policy_out[1].gather(1, action_batch).view(action_batch.shape[0],np.prod(action_batch.shape[1:]), 1,1)
    next_state_vector[non_final_mask, : ,:, :] = target_out[1].gather(1, target_out[0].max(1)[1].repeat(1,16).reshape((target_out[0].shape[0],1,4,4)))
    post_transaction_input_feature_map = next_state_vector[non_final_mask].view(target_out[1].shape[0], np.prod(target_out[1].shape[2:]), 1,1)

    generated_post_state = generator(post_transaction_input_feature_map)
    generated_pre_state = generator(pre_transaction_input_feature_map)

    original_stack = torch.cat([state_batch, non_final_next_states])
    generated_stack = torch.cat([generated_pre_state, generated_post_state])

    loss_g = F.mse_loss(original_stack, generated_stack)
    print("generator loss: \t" + str(loss_g.data.cpu().numpy()))
    # print("generator step: \t" + str(np.floor((np.log(1 - loss_g.data.cpu().numpy())/np.log(0.999)))))
    simulation_steps = np.floor((np.exp(1 - loss_g.data.cpu().numpy()) / np.exp(0.999)))
    print("generator step: \t" + str(simulation_steps))

    optimizer_g.zero_grad()
    loss_g.backward(retain_graph=True)
    for param in generator.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer_g.step()

    # policy optimization
    action_batch = torch.cat(batch.action)
    state_action_values = policy_out[0].gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_out[0].max(1)[0].detach()
    next_state_values = (next_state_values / (next_state_values.max(0)[0] - next_state_values.min(0)[0]))
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # run simulation
    n_step = 0
    while n_step < simulation_steps:
        target_out = target_net(generated_post_state)
        next_state_vector[non_final_mask, : ,:, :] = target_out[1].gather(1, target_out[0].max(1)[1].repeat(1,16).reshape((target_out[0].shape[0],1,4,4)))
        post_transaction_input_feature_map = next_state_vector[non_final_mask].view(target_out[1].shape[0], np.prod(target_out[1].shape[2:]), 1,1)
        generated_post_state = generator(post_transaction_input_feature_map)
        next_state_values[non_final_mask] = target_out[0].max(1)[0].detach()
        next_state_values = (next_state_values / (next_state_values.max(0)[0] - next_state_values.min(0)[0]))
        expected_state_action_values += (next_state_values * np.power(GAMMA, n_step))
        n_step += 1

    # losses = -expected_state_action_values.unsqueeze(1) * F.log_softmax(state_action_values, dim=-1)  # CE
    # loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    print("model loss: \t" + str(loss.data.cpu().numpy()))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()#retain_graph=True
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return generated_pre_state, generated_post_state, batch

# def optimize_generator():
#     if len(memory) < BATCH_SIZE:
#         return
#     # for l in range(BATCH_SIZE):
#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))
#     state_batch = torch.cat(batch.state)
#     policy_out = policy_net(state_batch)
#     action_batch = torch.cat(batch.action).repeat(1,16).reshape(32,1,4,4)
#     # input_feature_map = policy_out[1]#.gather(1, action_batch)
#     input_feature_map = policy_out[1].gather(1, action_batch).view(action_batch.shape[0],np.prod(action_batch.shape[1:]), 1,1)
#
#     generated_state = generator(input_feature_map)
#     # policy_generated_out = policy_net(generated_state)
#     # state_action_values = policy_generated_out[1] # .gather(1, action_batch)
#
#
#     # losses = -expected_state_action_values.unsqueeze(1) * F.log_softmax(state_action_values, dim=-1)  # CE
#     # loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch
#     # loss_g = F.smooth_l1_loss(state_action_values, policy_out[1])# expected_state_action_values.unsqueeze(1))
#     loss_g = F.mse_loss(state_batch, generated_state)
#     # Optimize the model
#     optimizer_g.zero_grad()
#     loss_g.backward()
#     # for param in policy_net.parameters():
#     #     param.grad.data.clamp_(-1, 1)
#     optimizer_g.step()
#     return generated_state, batch

# def optimize_generator():
#     if len(memory) < BATCH_SIZE:
#         return
#     # for l in range(BATCH_SIZE):
#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).cuda(device)
#     next_state_values = torch.zeros((BATCH_SIZE, 1 ,4,4), device=device)
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action).repeat(1,16).reshape(BATCH_SIZE,1,4,4)
#
#     policy_out = policy_net(state_batch)
#     pre_transaction_input_feature_map = policy_out[1].gather(1, action_batch).view(action_batch.shape[0],np.prod(action_batch.shape[1:]), 1,1)
#
#     target_out = target_net(non_final_next_states)
#     next_state_values[non_final_mask, : ,:, :] = target_out[1].gather(1, target_out[0].max(1)[1].repeat(1,16).reshape((target_out[0].shape[0],1,4,4)))
#     post_transaction_input_feature_map = next_state_values[non_final_mask].view(target_out[1].shape[0], np.prod(target_out[1].shape[2:]), 1,1)
#
#     generated_post_state = generator(post_transaction_input_feature_map)
#     generated_pre_state = generator(pre_transaction_input_feature_map)
#     # losses = -expected_state_action_values.unsqueeze(1) * F.log_softmax(state_action_values, dim=-1)  # CE
#     # loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch
#     # loss_g = F.smooth_l1_loss(state_action_values, policy_out[1])# expected_state_action_values.unsqueeze(1))
#     loss_g = F.mse_loss(non_final_next_states, generated_post_state)
#     # Optimize the model
#     optimizer_g.zero_grad()
#     loss_g.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer_g.step()
#
#     loss_g = F.mse_loss(state_batch, generated_pre_state)
#     # Optimize the model
#     optimizer_g.zero_grad()
#     loss_g.backward()
#     for param in policy_net.parameters():
#         param.grad.data.clamp_(-1, 1)
#     optimizer_g.step()
#     return generated_pre_state, generated_post_state, batch
######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#

num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    # state = current_screen
    state = current_screen - last_screen
    # state = (current_screen + last_screen)/2
    for t in count():
        start = time.time()
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        episode_reward.append(reward)
        end = time.time()
        # print("time 1: {:1.10f}".format(end - start))
        start = time.time()
        reward = torch.tensor([reward], device=device)

        # Observe new state

        last_screen = current_screen
        # current_screen = get_screen()
        current_screen = get_screen()*0.5 + last_screen*0.5
        if not done:
            # next_state = current_screen
            # next_state = (current_screen + last_screen)/2
            next_state = current_screen - last_screen
        else:
            next_state = None
        # Store the transition in memory
        memory.push(state, action, next_state, reward, t)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        end = time.time()
        # print("time 2: {:1.10f}".format(end - start))
        start = time.time()
        verbose = optimize_model()
        generated_pre_image, generated_post_image, batch = (None, None, None)
        if verbose:
            generated_pre_image, generated_post_image, batch = verbose
        end = time.time()
        # print("time 3: {:1.10f}".format(end - start))
        start = time.time()
        if done:
            episode_durations.append(t + 1)
            plot_durations(generated_pre_image, generated_post_image, batch)
            episode_rewards.append(np.mean(np.array(episode_reward)))
            episode_reward = []
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

######################################################################
# Here is the diagram that illustrates the overall resulting data flow.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# Actions are chosen either randomly or based on a policy, getting the next
# step sample from the gym environment. We record the results in the
# replay memory and also run optimization step on every iteration.
# Optimization picks a random batch from the replay memory to do training of the
# new policy. "Older" target_net is also used in optimization to compute the
# expected Q values; it is updated occasionally to keep it current.
#