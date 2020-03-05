import numpy as np
import gym
from dnc.dnc import DNC
import torch
import torch.optim as optim
from collections import namedtuple
import math
import random
import torch.nn.functional as F

from itertools import count
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
device = 0
GAMMA = 0.999
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

plt.ion()

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

steps_done = 0

def select_action(state, hidden, reset):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            hidden = repackage_hidden_dnc(hidden)
            x = state.unsqueeze(0).unsqueeze(0)
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            action, hidden = policy(x, hidden, reset_experience=reset).max(1)[1].view(1, 1)
            action = torch.tensor(action, dtype=torch.long)
            reset = False
            return action, hidden, reset
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), hidden


def optimize_model(hidden):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    x = state_batch.unsqueeze(0)
    x = x.type(torch.FloatTensor)
    x = x.cuda()
    hidden = repackage_hidden_dnc(hidden)
    state_action_values, _ = policy(x, hidden, reset_experience=reset)
    state_action_values = state_action_values.type(torch.LongTensor)
    state_action_values = state_action_values.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    print(str(optimizer.state_dict().keys()))
    for key in optimizer.state_dict()["state"].keys():
        print(str(key) + ":" + str(optimizer.state_dict()["state"][key].keys() )+ " " + str(optimizer.state_dict()["state"][key]["square_avg"].size()))
    print(str(optimizer.state_dict()["param_groups"][0]))

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(99)*means[0], means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

env = gym.make('CartPole-v0')
input_size = 4
n_actions = env.action_space.n
hidden = None

policy = DNC(4, 200, gpu_id=device, output_size=1)
target = DNC(4, 200, gpu_id=device, output_size=1)
target.load_state_dict(policy.state_dict())
target.eval()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(policy.parameters())
observation = torch.from_numpy(env.reset())
episode_number = 0
reward_sum = 0
reset = False
episode_durations = []

while episode_number <= 5000:
    for t in count():
        env.render()
        x = observation.unsqueeze(0).unsqueeze(0)
        x = x.type(torch.FloatTensor)
        x = x.cuda()
        hidden = repackage_hidden_dnc(hidden)
        left_prob, hidden = policy(x, hidden, reset_experience=reset)
        reset = False
        action = 1 if np.random.uniform() < left_prob.item() else 0
        # record various intermediates (needed later for backprop)
        next_observation, reward, done, info = env.step(action)
        if not isinstance(next_observation, torch.Tensor):
            next_observation = torch.from_numpy(next_observation)
        if not isinstance(observation, torch.Tensor):
            observation = torch.from_numpy(observation)
        if not isinstance(reward, torch.Tensor):
            reward = torch.from_numpy(np.ones(1)*reward)
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor(np.ones(1)*action)

        # next_observation = next_observation.cuda()
        if done:
            next_observation = None
        memory.push(observation, action, next_observation, reward)
        observation = next_observation

        optimize_model(hidden)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            episode_number += 1
            observation = torch.from_numpy(env.reset())
            reset = True
            break





