
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
from utils import elapsed, build_dataset

class LSTM:
    n_input = 0
    n_hidden = 0
    weights = {}
    biases = {}
    pred = None
    cost = 0
    correct_pred = None
    learning_rate = 0
    writer = None
    logs_path = '/tmp/tensorflow/rnn_words'
    n_output = 0
    optimizer = None
    accuracy = 0
    display_step = -1
    dictionary = {}
    reverse_dictionary = {}
    training_data = None

    def __init__(self, training_data,n_input, n_hidden, learning_rate, verbose=None):
        self.training_data = training_data
        self.dictionary,  self.reverse_dictionary = build_dataset(self.training_data)
        if verbose:
            # Target log path
            self.writer = tf.summary.FileWriter(self.logs_path)
            self.display_step = 1000
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = len(self.dictionary)
        self.x = tf.placeholder("float", [None, n_input, 1])
        self.y = tf.placeholder("float", [None,self. n_output])
        # RNN output node weights and biases
        self.weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, self.n_output]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }
        self.pred = self.RNN()
        self.learning_rate = learning_rate
        # Loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Model evaluation
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

    def RNN(self):
            # reshape to [1, n_input]
            x = tf.reshape(self.x, [-1, self.n_input])

            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            x = tf.split(x, self.n_input, 1)

            # 2-layer LSTM, each layer has n_hidden units.
            # Average Accuracy= 95.20% at 50k iter
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])

            # 1-layer LSTM with n_hidden units but with lower accuracy.
            # Average Accuracy= 90.60% 50k iter
            # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
            # rnn_cell = rnn.BasicLSTMCell(n_hidden)

            # generate prediction
            outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

            # there are n_input outputs but
            # we only want the last output
            return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def train(self, max_training_iters):
        # self._merge_dictionaries(self.dictionary)
        start_time = time.time()
        # Launch the graph
        with tf.Session() as session:
            session.run(self.init)
            step = 0
            offset = random.randint(0,self.n_input+1)
            end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0
            if self.writer:
                self.writer.add_graph(session.graph)
            saver = tf.train.Saver()

            while step < max_training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                if offset > (len(self.training_data)-end_offset):
                    offset = random.randint(0, self.n_input+1)

                symbols_in_keys = [[self.dictionary[ str(self.training_data[i])]] for i in range(offset, offset+self.n_input) ]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

                symbols_out_onehot = np.zeros([self.n_output], dtype=float)
                symbols_out_onehot[self.dictionary[str(self.training_data[offset+self.n_input])]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

                _, acc, loss, onehot_pred = session.run([self.optimizer, self.accuracy, self.cost, self.pred], feed_dict={self.x: symbols_in_keys, self.y: symbols_out_onehot, self.learning_rate: 0.5})
                loss_total += loss
                acc_total += acc
                if self.display_step > 0 and (step+1) % self.display_step == 0 :
                    print("Iter= " + str(step+1) + ", Average Loss= " + \
                          "{:.6f}".format(loss_total/self.display_step) + ", Average Accuracy= " + \
                          "{:.2f}%".format(100*acc_total/self.display_step))
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [self.training_data[i] for i in range(offset, offset + self.n_input)]
                    symbols_out = self.training_data[offset + self.n_input]
                    symbols_out_pred = self.reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                    print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
                step += 1
                offset += (self.n_input+1)
            print("Optimization Finished!")
            for v in tf.all_variables():
                print(v.name)
            print("Elapsed time: ", elapsed(time.time() - start_time))
            saver.save(session, './model.ckpt')
        tf.reset_default_graph()

    def predict(self, sentence, default_graph):
        tf.import_graph_def(default_graph)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != self.n_input:
            return "Need " + str(self.n_input) + " words"
        try:
            with tf.Session() as session:
                symbols_in_keys = [self.dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])
                    onehot_pred = session.run(self.pred, feed_dict={self.x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,self.reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                return sentence
        except KeyError:
            return "Word not in dictionary"

    def _merge_dictionaries(self, dictionary):
        for word in dictionary.keys():
            if word in self.dictionary.keys():
                self.dictionary[word] += dictionary[word]
            else:
                self.dictionary[word] = dictionary[word]
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
