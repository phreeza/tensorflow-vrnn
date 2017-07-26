from utils import create_dir, pickle_save
from config import SAVE_DIR, VRNNConfig
from datetime import datetime
from model import VRNNCell

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import pickle
import os

logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VRNN(VRNNConfig):
    def __init__(self, istest=False):
        VRNNConfig.__init__(self)

        def tf_normal(y, mu, sigma):
            with tf.variable_scope('normal'):
                sigma_square = tf.maximum(1e-10, tf.square(sigma))
                norm = tf.subtract(y[:,:args.chunk_samples], mu)
                z = tf.div(tf.square(norm), sigma_square)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                result = tf.reduce_sum(z+denom_log, 1)/2#
            return result

        def kl_gaussian(mu_1, sigma_1, mu_2, sigma_2):
            with tf.variable_scope("kl_gaussisan"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9, sigma_2),name='log_sigma_2')
                  - 2 * tf.log(tf.maximum(1e-9, sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma, y):
            kl_loss = kl_gaussian(enc_mu, enc_sigma, prior_mu, prior_sigma) # KL_divergence
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma)
            return tf.reduce_mean(kl_loss + likelihood_loss)

        if istest:
            self.batch_size = 1
            self.seq_length = 1

        cell = VRNNCell(self.chunk_samples, self.rnn_size, self.latent_size)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length, 2*self.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length, 2*self.chunk_samples], name = 'target_data')
        self.initial_state_c, self.initial_state_h = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # [n_steps, batch_size, n_input]
            inputs = tf.reshape(inputs, [-1, 2*self.chunk_samples]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(axis=0, num_or_size_splits=self.seq_length, value=inputs) # n_steps * (batch_size, n_hidden)
        flat_target_data = tf.reshape(self.target_data, [-1, 2*self.chunk_samples])

        self.target = flat_target_data
        self.flat_input = tf.reshape(tf.transpose(tf.stack(inputs), [1,0,2]), [self.batch_size*self.seq_length, -1])
        self.input = tf.stack(inputs)
        # Get vrnn cell output
        outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=(self.initial_state_c, self.initial_state_h))
        #print outputs
        #outputs = map(tf.pack,zip(*outputs))
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "prior_mu", "prior_sigma"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[self.batch_size*self.seq_length, -1])
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma = outputs_reshape
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma

        self.cost = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_sigma, prior_mu, prior_sigma, flat_target_data)
        self.sigma = dec_sigma
        self.mu = dec_mu

        print_vars("trainable_variables")
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        sess = tf.Session()

    def sample(self, num=4410, start=None):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape))

        if start is None:
            prev_x = np.random.randn(1, 1, 2*self.chunk_samples)
        elif len(start.shape) == 1:
            prev_x = start[np.newaxis,np.newaxis,:]
        elif len(start.shape) == 2:
            for i in range(start.shape[0]-1):
                prev_x = start[i,:]
                prev_x = prev_x[np.newaxis,np.newaxis,:]
                feed = {self.input_data: prev_x,
                        self.initial_state_c:prev_state[0],
                        self.initial_state_h:prev_state[1]}

                [o_mu, o_sigma, o_rho, prev_state_c, prev_state_h] = sess.run(
                        [self.mu, self.sigma, self.rho,
                         self.final_state_c,self.final_state_h],feed)

            prev_x = start[-1,:]
            prev_x = prev_x[np.newaxis,np.newaxis,:]

        prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        chunks = np.zeros((num, 2*self.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, self.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, self.chunk_samples), dtype=np.float32)

        for i in xrange(num):
            feed = {self.input_data: prev_x,
                    self.initial_state_c:prev_state[0],
                    self.initial_state_h:prev_state[1]}
            [o_mu, o_sigma, o_rho, next_state_c, next_state_h] = sess.run([self.mu, self.sigma,
                self.rho, self.final_state_c, self.final_state_h],feed)

            next_x = np.hstack((sample_gaussian(o_mu, o_sigma),
                                2.*(o_rho > np.random.random(o_rho.shape[:2]))-1.))
            chunks[i] = next_x
            mus[i] = o_mu
            sigmas[i] = o_sigma

            prev_x = np.zeros((1, 1, 2*self.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            prev_state = next_state_c, next_state_h

        return chunks, mus, sigmas

    def next_batch(self):
        t_offset = np.random.randn(self.batch_size, 1, (2 * self.chunk_samples))
        mixed_noise = np.random.randn(
            self.batch_size, self.seq_length, (2 * self.chunk_samples)) * 0.1
        x = np.random.randn(self.batch_size, self.seq_length, (2 * self.chunk_samples)) * 0.1
            + mixed_noise*0.1
            + np.sin(2 * np.pi * (np.arange(self.seq_length)[np.newaxis, :, np.newaxis] / 10. + t_offset))

        y = np.random.randn(self.batch_size, self.seq_length, (2 * self.chunk_samples)) * 0.1
            + mixed_noise*0.1
            + np.sin(2 * np.pi * (np.arange(1, self.seq_length+1)[np.newaxis, :, np.newaxis] / 10. + t0))
        y[:, :, self.chunk_samples:] = 0.
        x[:, :, self.chunk_samples:] = 0.
        return x, y

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        self.n_batches = 100
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        saver = tf.train.Saver(tf.global_variables())

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Loaded model")

        for epoch in range(self.num_epochs):
            # Learning rate decay
            sess.run(tf.assign(model.lr, self.learning_rate * (self.decay_rate ** epoch)))

            for b in range(self.n_batches):
                x, y = next_batch(args)
                feed_dict = {model.input_data: x, model.target_data: y}
                train_loss, _, cr, sigma= sess.run([model.cost, model.train_op, check, model.sigma], feed_dict = feed_dict)

                if (e * self.n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * self.n_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                print("{}/{}(epoch {}), train_loss = {:.6f}, std = {:.3f}".format(e * self.n_batches + b, args.num_epochs * n_batches, e, self.chunk_samples * train_loss, sigma.mean(axis=0).mean(axis=0)))

if __name__ == '__main__':
    model = VRNN()
    model.initialize()
    model.train()
