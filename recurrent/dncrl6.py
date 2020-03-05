
import random
import gym
import math
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, LSTMCell, Dense, LSTM, RNN, Conv2D, MaxPooling2D, Reshape, Flatten, Input, TimeDistributed, GRU, Permute, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
# from ntm.ntm import NeuralTuringMachine as NTM
from tfDNC.dnc.dnc import DNC
import tqdm
import matplotlib.pyplot as plt

class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, reccurrent_size=4, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.001, alpha_decay=0.01, batch_size=1, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        self.train = True
        self.reccurrent_size = reccurrent_size
        self.episode_durations = []

        plt.ion()
        plt.figure()
        plt.show()
        # self.resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps


        self.adapter = Sequential()
        self.adapter.add(TimeDistributed(Flatten(input_shape=(None, 4)), input_shape=(self.reccurrent_size, 4), batch_size=batch_size))
        # self.adapter.add(Dense(4, batch_size=batch_size))
        # self.adapter.add(Dense(48, batch_size=batch_size))
        # self.adapter.add(Dense(24, batch_size=batch_size))

        memory_config = {
            'memory_size': 32,
            'word_size': 9,
            'num_read_heads': 10,
        }

        main_input = Input(shape=(self.reccurrent_size, 4), batch_size=batch_size)#(None, 32, ))
        self.dnc_cell = DNC(2, controller_units=30, **memory_config)
        dnc_initial_state = self.dnc_cell.get_initial_state(batch_size=batch_size)
        layer = RNN(self.dnc_cell)
        self.adapter.add(layer)
        self.adapter = self.adapter(main_input)
        # layer = layer(self.adapter, dnc_initial_state)
        self.model = Model(main_input, self.adapter)
        self.model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', run_eagerly=True)
        self.model.summary()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, epsilon):
        time_lapse = None
        if np.random.random() <= epsilon or len(self.memory) < self.reccurrent_size:
            sampled = self.env.action_space.sample()
        else:
            time_lapse = []
            for i in range(len(self.memory) - self.reccurrent_size, len(self.memory)):
                state, action, reward, next_state, done = self.memory[i]
                time_lapse.append(state)
            time_lapse = np.array(time_lapse)
            sampled = np.argmax(self.model.predict(np.expand_dims(time_lapse, axis=0)).reshape([-1, 1, 2]))
        return sampled, time_lapse

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def replay(self, batch_size):
        if len(self.memory) < self.reccurrent_size:
            return

        x_batch, y_batch, y_targets = [], [], []

        minibatchi = random.sample(range(len(self.memory)), min(len(self.memory), batch_size))
        mini_curr_batch = []
        mini_next_batch = []
        for sample in minibatchi:
            if sample > self.reccurrent_size:
                mini_mini_curr_batch = []
                mini_mini_next_batch = []
                for t in range(self.reccurrent_size):
                    mini_mini_curr_batch.append(self.memory[sample - self.reccurrent_size + t][0])
                    mini_mini_next_batch.append(self.memory[sample - self.reccurrent_size + t])
                mini_curr_batch.append(np.array(mini_mini_curr_batch))
                mini_next_batch.append(np.array(mini_mini_next_batch))
        if not mini_curr_batch or not mini_next_batch:
            return
        for time_lapse1, time_lapse2 in zip(mini_curr_batch, mini_next_batch):
            time_lapse1 = tf.cast(time_lapse1, tf.float32)
            y_target = self.model.predict(np.expand_dims(time_lapse1, axis=0))
            state, action, reward, next_state, done = time_lapse2[-1]
            time_lapse2 = np.expand_dims(np.array([np.array(x) for x in time_lapse2[:, 3]]), axis=0)
            time_lapse2 = tf.cast(time_lapse2, tf.float32)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(time_lapse2)[0])
            x_batch.append(time_lapse1)
            y_batch.append(y_target[0])

        y_batch = np.array(y_batch)
        x_batch = np.array(x_batch)
        self.model.fit(x_batch, y_batch, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)
        state = self.env.reset()
        for e in tqdm.tqdm(range(self.n_episodes)):
            done = False
            i = 0
            while not done:
                # if not self.train:
                self.env.render()
                action, time_lapse = self.choose_action(self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                # next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
            state = self.env.reset()
            self.episode_durations.append(i)
            scores.append(i)
            # self.dnc_cell.state = None
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
            if self.train:
                self.replay(self.batch_size)
            self.plot_durations()
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array')#.transpose((2, 0, 1))
        # # Cart is in the lower half, so strip off the top and bottom of the screen
        screen_height, screen_width, _ = screen.shape
        screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:,:]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, slice_range, :]
        # # Convert to float, rescale, convert to torch tensor
        # # (this doesn't require a copy)
        # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # # screen = torch.from_numpy(screen)
        # # Resize, and add a batch dimension (BCHW)
        return np.reshape(screen, (screen.shape[0], screen.shape[1], screen.shape[2]))  # resize(screen).unsqueeze(0).to(device)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.episode_durations)
        plt.pause(0.001)  # pause a bit so that plots are updated
if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()