from ops import fc_layer, get_shape, print_vars
import tensorflow as tf
import numpy as np

class VRNNCell(tf.nn.rnn_cell.RNNCell):
    """Variational RNN cell."""

    def __init__(self, x_dim, h_dim, z_dim = 100):
        '''
        Args:
            x_dim - chunk_samples
            h_dim - rnn_size
            z_dim - latent_size
        '''
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.n_h, state_is_tuple=True)

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, x, state, scope=None):
        '''
		Args:
			x - input 2D tensor [batch_size x 2*self.chunk_samples]
			state - tuple
				(hidden, cell_state)
			scope - string
				defaults to be None
    	'''
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state
            with tf.variable_scope("Prior"):
                prior_hidden = fc_layer(h, self.n_prior_hidden, activation = tf.nn.relu, scope = "hidden")
                prior_mu = fc_layer(prior_hidden, self.n_z, scope = "mu")
                prior_sigma = fc_layer(prior_hidden, self.n_z, activation = tf.nn.softplus, scope = "sigma")# >=0

            x_1 = fc_layer(x, self.n_x_1, activation = tf.nn.relu, scope = "phi_x")# >=0

            with tf.variable_scope("Encoder"):
                enc_hidden = fc_layer(tf.concat(values=(x_1, h), axis=1), self.n_enc_hidden, activation = tf.nn.relu, scope = "hidden")
                enc_mu = fc_layer(enc_hidden, self.n_z, scope = 'mu')
                enc_sigma = fc_layer(enc_hidden, self.n_z, activation = tf.nn.softplus, scope = 'sigma')

            # Random sampling ~ N(0, 1)
            eps = tf.random_normal((get_shape(x)[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon, latent variable from reparametrization trick
            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))
            z_1 = fc_layer(z, self.n_z_1, activation = tf.nn.relu, scope = "phi_z")

            with tf.variable_scope("Decoder"):
                dec_hidden = fc_layer(tf.concat(values=(z_1, h), axis=1), self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden")
                dec_mu = fc_layer(dec_hidden, self.n_x, scope = "mu")
                dec_sigma = fc_layer(dec_hidden, self.n_x, activation = tf.nn.softplus, scope = "sigma")

            output, next_state = self.lstm(tf.concat(values=(x_1, z_1), axis=1), state)

        return (enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma), next_state
