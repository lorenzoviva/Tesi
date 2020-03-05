
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import LSTMCell, Dense, LSTM, Conv2D, MaxPooling2D, Reshape, Flatten, Input, TimeDistributed, GRU, Permute
from keras.optimizers import Adam, RMSprop
from dnc.dnc import DNC
import tqdm
import matplotlib.pyplot as plt
class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, reccurrent_size=2, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.001, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
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

        # Init model
        self.model = Sequential() # DNC(4, 24, output_size=2)
        # self.model.add(Dense(24, input_dim=4, activation='tanh'))
        # dnc = DNC(4, 24, output_size=2)
        # for rnn in dnc.rnns:
        #     self.model.add(rnn)
        # self.model.add(Dense(48,  activation='tanh'))
        # self.model.add(Dense(24, activation='tanh', input_shape=(1, 4)))
        # self.model.add(Dense(48, activation='tanh'))
        # self.model.add(Conv2D(16, 5, strides=2, input_shape=(160, 360, 3)))
        # self.model.add(BatchNormalization(input_shape=(16)))
        self.cnn = Sequential()
        self.cnn.add(MaxPooling2D(pool_size=(4, 4)))
        self.cnn.add(Conv2D(8, (3, 3), input_shape=(100, 150, 3), data_format="channels_last", padding="same"))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Conv2D(8, (3, 3), input_shape=(50, 75, 3), data_format="channels_last", padding="same"))
        # self.cnn.add(MaxPooling2D(pool_size=(4, 4)))
        self.cnn.add(Flatten(data_format="channels_last")) # Not sure if this if the proper way to do this.
        # self.cnn.add(Reshape((-1,100*150))) # Not sure if this if the proper way to do this.
        # self.cnn.add(Permute((2, 1), input_shape=(-1,8,150,100)))

        self.rnn = Sequential()
        self.rnn.add(GRU(8, return_sequences=False, input_shape=(8, 30000)))  # 60
        # self.rnn.add((Dense(256)))

        self.dense = Sequential()
        self.dense.add(Dense(128))
        self.dense.add(Dense(64))
        self.dense.add(Dense(2)) # Model output

        main_input = Input(shape=(self.reccurrent_size, 400, 600, 3)) # Data has been reshaped to (800, 5, 120, 60, 1)
        # target_input = Input(shape=(self.reccurrent_size, 400, 600, 3))
        model = TimeDistributed(self.cnn)(main_input) # this should make the cnn 'run' 5 times?
        model = self.rnn(model) # combine timedistributed cnn with rnn
        model = self.dense(model) # add dense
        self.model = Model(inputs=main_input, outputs=model)
        # self.target = Model(inputs=target_input, outputs=model)

        # self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, screen_state, state, action, reward, next_screen_state, next_state, done):
        self.memory.append((screen_state, state, action, reward, next_screen_state, next_state, done))

    def choose_action(self, epsilon):
        time_lapse = None
        if np.random.random() <= epsilon or len(self.memory) < self.reccurrent_size:
            sampled = self.env.action_space.sample()
        else:
            time_lapse = []
            for i in range(len(self.memory) - self.reccurrent_size, len(self.memory)):
                screen_state, state, action, reward, next_screen_state, next_state, done = self.memory[i]
                time_lapse.append(screen_state)
            time_lapse = np.array(time_lapse)
            sampled = np.argmax(self.model.predict(np.expand_dims(time_lapse, axis=0)).reshape([-1, 1, 2]))
        return sampled, time_lapse

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    # def preprocess_state(self, state):
    #     state = np.reshape(state, [1, 4])
    #     return self.prepare_for_LSTM(state)
    #
    # def prepare_for_LSTM(self, data):
    #     return data.reshape(data.shape[0], 1, data.shape[1])
    #
    # def after_LSTM(self, data):
    #     return data.reshape([1,2])

    def replay(self, batch_size):
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
        for time_lapse1, time_lapse2 in zip(mini_curr_batch, mini_next_batch):
            y_target = self.model.predict(np.expand_dims(time_lapse1, axis=0))
            screen_state, state, action, reward, next_screen_state, next_state, done = time_lapse2[-1]
            time_lapse2 = np.expand_dims(np.array([np.array(x) for x in time_lapse2[:, 4]]), axis=0)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(time_lapse2)[0])
            x_batch.append(time_lapse1)
            y_batch.append(y_target[0])

        y_batch = np.array(y_batch)
        x_batch = np.array(x_batch)
        self.model.fit(np.array(x_batch), y_batch, batch_size=len(x_batch), verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)
        state = self.env.reset()
        last_screen = self.get_screen()
        current_screen = self.get_screen()
        screen_state = current_screen - last_screen
        for e in tqdm.tqdm(range(self.n_episodes)):
            done = False
            i = 0
            while not done:
                # if not self.train:
                self.env.render()
                action, time_lapse = self.choose_action(self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                # next_state = self.preprocess_state(next_state)
                last_screen = current_screen
                current_screen = self.get_screen()
                next_screen_state = current_screen - last_screen
                self.remember(screen_state, state, action, reward, next_screen_state, next_state, done)
                state = next_state
                i += 1
            state = self.env.reset()
            self.episode_durations.append(i)
            scores.append(i)
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
        # screen_height, screen_width, _ = screen.shape
        # screen = screen[int(screen_height*0.4):int(screen_height * 0.8),:,:]
        # view_width = int(screen_width * 0.6)
        # cart_location = self.get_cart_location(screen_width)
        # if cart_location < view_width // 2:
        #     slice_range = slice(view_width)
        # elif cart_location > (screen_width - view_width // 2):
        #     slice_range = slice(-view_width, None)
        # else:
        #     slice_range = slice(cart_location - view_width // 2,
        #                         cart_location + view_width // 2)
        # # Strip off the edges, so that we have a square image centered on a cart
        # screen = screen[slice_range, :, :]
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