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

from RL_ACR4 import *


common_nn_size=[128,64,32,16,8,4,2]
init_caps_size=[128,64,32,16,8,4,2]
init_caps_number=[128,64,32,16,8,4,2]
routing_layers=[[],[(2,2)], [(2,4)], [(4,8)], [(8,16)], [(16,32)], [(4,32)], [(32,4)], [(2,4),(4,8)], [(4,32),(32,16)], [(64,4),(16,8)],[(4,4),(8,8)], [(16,16),(16,16)], [(2,2),(2,2)]]
# for common_nn_siz in common_nn_size: #128
#     for init_caps_siz in init_caps_size: #16
#         for init_caps_numbe in init_caps_number: # 8
#             for routing_layer in routing_layers: # [(2,4),(4,8)],
#                 main(common_nn_siz, init_caps_siz, init_caps_numbe, routing_layer)

num_trials = np.prod([len(choices) for choices in [common_nn_size, init_caps_size, init_caps_number, routing_layers]])

def trial(number):
    n = 0
    for common_nn_siz in common_nn_size: #128
        for init_caps_siz in init_caps_size: #16
            for init_caps_numbe in init_caps_number: # 8
                for routing_layer in routing_layers: # [(2,4),(4,8)],
                    if n == number:
                        return common_nn_siz, init_caps_siz, init_caps_numbe, routing_layer
                    n += 1



i = 2286
while i < num_trials:
    common_nn_siz,init_caps_siz,init_caps_numbe,routing_layer = trial(i)
    main(common_nn_siz, init_caps_siz, init_caps_numbe, routing_layer)
    i += 1