from ops import fc_layer, get_shape, print_vars
import tensorflow as tf
import numpy as np

class VartiationalRNNCell(tf.nn.rnn_cell.RNNCell):
    """Variational RNN cell."""

    def __init__(self, x_dim, h_dim, z_dim = 100):
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
			x - input 2D tensor
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
                dec_hidden = fc_layer(tf.concat(values=(z1, h), axis=1), self.n_dec_hidden, activation = tf.nn.relu, scope = "hidden")
                dec_mu = fc_layer(dec_hidden, self.n_x, scope = "mu")
                dec_sigma = fc_layer(dec_hidden, self.n_x, scope = "sigma")

            output, next_state = self.lstm(tf.concat(values=(x_1, z_1), axis=1), state)
        return (enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma), next_state

class VRNN():
    def __init__(self, args, istest=False):
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

        self.args = args
        
        if istest:
            args.batch_size = 1
            args.seq_length = 1

        cell = VartiationalRNNCell(args.chunk_samples, args.rnn_size, args.latent_size)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name = 'target_data')
        self.initial_state_c, self.initial_state_h = self.cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # [n_steps, batch_size, n_input]
            inputs = tf.reshape(inputs, [-1, 2*args.chunk_samples]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(axis=0, num_or_size_splits=args.seq_length, value=inputs) # n_steps * (batch_size, n_hidden)
        flat_target_data = tf.reshape(self.target_data, [-1, 2*args.chunk_samples])

        self.target = flat_target_data
        self.flat_input = tf.reshape(tf.transpose(tf.stack(inputs), [1,0,2]), [args.batch_size*args.seq_length, -1])
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
                x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
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

    def sample(self, sess, args, num=4410, start=None):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape))

        if start is None:
            prev_x = np.random.randn(1, 1, 2*args.chunk_samples)
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
        chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)

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

            prev_x = np.zeros((1, 1, 2*args.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            prev_state = next_state_c, next_state_h

        return chunks, mus, sigmas
