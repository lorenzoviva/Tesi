
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import LSTMCell, Dense, LSTM, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from dnc.dnc import DNC
import tqdm

class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.001, batch_size=64, monitor=False, quiet=False):
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

        # self.model.add(Conv2D(24, (3, 3), padding="same", activation="relu", input_shape=(3, 180, 16)))
        self.model.add(Conv2D(48, (3, 3), padding="same", activation="relu", data_format="channels_first", input_shape=(3, 160, 360)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.model.add(Conv2D(24, (3, 3), padding="same", activation="relu", data_format="channels_first" ))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.model.add(Conv2D(16, (3, 3), padding="same", activation="relu", data_format="channels_first"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.model.add(Reshape((1, 14400)))
        self.model.add(Dense(16, input_shape=(1, 14400)))
        self.model.add(LSTM(2,  input_shape=(1, 16), return_sequences=True))
        self.model.add(Reshape((2, )))

        # self.model.add(Activation("softmax"))

        # self.model.add(Conv2D(16, 5, strides=2, input_shape=(5, 180, 16)))
        # self.model.add(Dense(2, activation='linear'))

    # self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, screen_state, state, action, reward, next_screen_state, next_state, done):
        self.memory.append((screen_state, state, action, reward, next_screen_state, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            sampled = self.env.action_space.sample()
        else:
            sampled = np.argmax(self.model.predict(state).reshape([1, 2]))
        return sampled

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
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for screen_state, state, action, reward, next_screen_state, next_state, done in minibatch:
            y_target = self.model.predict(screen_state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_screen_state)[0])
            x_batch.append(screen_state[0])
            y_batch.append(y_target[0])
        y_batch = np.array(y_batch)
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
                if not self.train:
                    self.env.render()
                action = self.choose_action(screen_state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                # next_state = self.preprocess_state(next_state)
                last_screen = current_screen
                current_screen = self.get_screen()
                next_screen_state = current_screen - last_screen
                self.remember(screen_state, state, action, reward, next_screen_state, next_state, done)
                state = next_state
            i += 1
            state = self.env.reset()

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
            if self.train:
                self.replay(self.batch_size)

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
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
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return np.reshape(screen, (1, screen.shape[0], screen.shape[1], screen.shape[2]))  # resize(screen).unsqueeze(0).to(device)

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()