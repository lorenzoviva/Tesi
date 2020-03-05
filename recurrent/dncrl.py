
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import LSTMCell, Dense, LSTM, Conv2D
from keras.optimizers import Adam
from dnc.dnc import DNC


class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.003, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
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
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential() # DNC(4, 24, output_size=2)
        # self.model.add(Dense(24, input_dim=4, activation='tanh'))
        # dnc = DNC(4, 24, output_size=2)
        # for rnn in dnc.rnns:
        #     self.model.add(rnn)
        # self.model.add(Dense(48,  activation='tanh'))
        self.model.add(Dense(24, activation='tanh', input_shape=(1, 4)))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(LSTM(2, input_shape=(1, 2),  return_sequences=True))
        # self.model.add(Dense(2, activation='linear'))

    # self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            sampled = self.env.action_space.sample()
        else:
            sampled = np.argmax(self.model.predict(state).reshape([1, 2]))
        return sampled

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        state = np.reshape(state, [1, 4])
        return self.prepare_for_LSTM(state)

    def prepare_for_LSTM(self, data):
        return data.reshape(data.shape[0], 1, data.shape[1])

    def after_LSTM(self, data):
        return data.reshape([1,2])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.after_LSTM(self.model.predict(state))
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        y_batch = np.array(y_batch)
        y_batch = self.prepare_for_LSTM(y_batch)
        self.model.fit(np.array(x_batch), y_batch, batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()