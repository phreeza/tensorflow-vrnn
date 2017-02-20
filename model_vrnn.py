import tensorflow as tf
import numpy as np

#Todos:
# - Add batch normalization
# - Add KL term annealing
# - try fixed variance/vanilla least squares in output layer

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class VartiationalRNNCell(tf.nn.rnn_cell.RNNCell):
    """Variational RNN cell."""

    def __init__(self, x_dim, h_dim, z_dim = 100):
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = 2*z_dim
        self.n_dec_hidden = 2*x_dim
        self.n_prior_hidden = 2*z_dim
        self.layers_enc_hidden = 5
        self.layers_dec_hidden = 5
        self.layers_prior_hidden = 5
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.n_h, state_is_tuple=True)


    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            with tf.variable_scope("Prior"):
                with tf.variable_scope("hidden"):
                    prev = h
                    for n in range(self.layers_prior_hidden):
                        with tf.variable_scope("Layer_"+str(n+1)):
                            prev = tf.nn.relu(linear(prev, self.n_prior_hidden))
                    prior_hidden = prev
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))

            with tf.variable_scope("phi_x"):
                x_1 = tf.nn.relu(linear(x, self.n_x_1))

            with tf.variable_scope("Encoder"):
                with tf.variable_scope("hidden"):
                    prev = tf.concat(1,(x_1, h))
                    for n in range(self.layers_enc_hidden):
                        with tf.variable_scope("Layer_"+str(n+1)):
                            prev = tf.nn.relu(linear(prev, self.n_enc_hidden))
                    enc_hidden = prev
                with tf.variable_scope("mu"):
                    enc_mu    = linear(enc_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.nn.softplus(linear(enc_hidden, self.n_z))
            eps = tf.random_normal((x.get_shape().as_list()[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z = tf.add(enc_mu, tf.mul(enc_sigma, eps))
            with tf.variable_scope("phi_z"):
                z_1 = tf.nn.relu(linear(z, self.n_z_1))

            with tf.variable_scope("Decoder"):
                with tf.variable_scope("hidden"):
                    prev = tf.concat(1,(z_1, h))
                    for n in range(self.layers_dec_hidden):
                        with tf.variable_scope("Layer_"+str(n+1)):
                            prev = tf.nn.relu(linear(prev, self.n_dec_hidden))
                    dec_hidden = prev
                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden, self.n_x)
                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden, self.n_x))
                with tf.variable_scope("rho"):
                    dec_rho = tf.nn.sigmoid(linear(dec_hidden, self.n_x))


            output, state2 = self.lstm(tf.concat(1,(x_1, z_1)), state)
        return (enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma), state2




class VRNN():
    def __init__(self, args, sample=False):

        def tf_normal(y, mu, s, rho):
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10,tf.square(s))
                norm = tf.sub(y[:,:args.chunk_samples], mu)
                z = tf.div(tf.square(norm), ss)
                signal_sign = y[:,args.chunk_samples:]
                denom_log = tf.log(2.*np.pi*ss, name='denom_log')
                result = tf.reduce_sum((z+denom_log)/2. -
                                       (tf.log(tf.maximum(1e-20,rho),name='log_rho')*(1.+signal_sign)
                                        +tf.log(tf.maximum(1e-20,1.-rho),name='log_rho_inv')*(1.-signal_sign))/2., 1)

            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                    2. * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
                  - 2. * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma, y):
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma, dec_rho)

            return tf.reduce_mean(kl_loss + likelihood_loss), tf.reduce_mean(kl_loss), tf.reduce_mean(likelihood_loss)
            #return tf.reduce_mean(likelihood_loss)

        self.args = args
        if sample:
            args.batch_size = 1
            args.seq_length = 1

        cell = VartiationalRNNCell(args.chunk_samples, args.rnn_size, args.latent_size)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples],name = 'target_data')
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)


        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, 2*args.chunk_samples]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(0, args.seq_length, inputs) # n_steps * (batch_size, n_hidden)
        flat_target_data = tf.reshape(self.target_data,[-1, 2*args.chunk_samples])

        self.target = flat_target_data
        self.flat_input = tf.reshape(tf.transpose(tf.pack(inputs),[1,0,2]),[args.batch_size*args.seq_length, -1])
        self.input = tf.pack(inputs)
        # Get vrnn cell output
        outputs, last_state = tf.nn.rnn(cell, inputs, initial_state=(self.initial_state_c,self.initial_state_h))
        #print outputs
        #outputs = map(tf.pack,zip(*outputs))
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_rho", "prior_mu", "prior_sigma"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.pack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma = outputs_reshape
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.rho = dec_rho

        lossfunc, kl_loss, likelihood_loss = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma, flat_target_data)
        with tf.variable_scope('cost'):
            self.cost = lossfunc 
            self.kl_loss = kl_loss
            self.likelihood_loss = likelihood_loss
        with tf.variable_scope('avg_stats'):
            self.mu_avg = tf.reduce_mean(dec_mu,name='mu_avg')
            self.sigma_avg = tf.reduce_sum(dec_sigma,name='sigma_avg')/(args.batch_size*args.seq_length*args.chunk_samples)
            self.rho_avg = tf.reduce_mean(dec_rho,name='rho_avg')

        tf.scalar_summary('cost', self.cost)
        tf.scalar_summary('kl_loss', self.kl_loss)
        tf.scalar_summary('likelihood_loss', self.likelihood_loss)
        tf.scalar_summary('mu', self.mu_avg)
        tf.scalar_summary('sigma', self.sigma_avg)
        tf.scalar_summary('rho', self.rho_avg)


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
        grads = tf.gradients(self.cost, tvars)
        #grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self.saver = tf.train.Saver(tf.all_variables())

    def sample(self, sess, args, num=4410, start=None, T=1.):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape))

        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

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

                prev_state = prev_state_c, prev_state_h

            prev_x = start[-1,:]
            prev_x = prev_x[np.newaxis,np.newaxis,:]

        chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)
        rhos = np.zeros((num, args.chunk_samples), dtype=np.float32)

        for i in xrange(num):
            feed = {self.input_data: prev_x,
                    self.initial_state_c:prev_state[0],
                    self.initial_state_h:prev_state[1]}
            [o_mu, o_sigma, o_rho, next_state_c, next_state_h] = sess.run([self.mu, self.sigma,
                self.rho, self.final_state_c, self.final_state_h],feed)

            next_x = np.hstack((sample_gaussian(o_mu, o_sigma*T),
                                2.*(o_rho > np.random.random(o_rho.shape[:2]))-1.))
            chunks[i] = next_x
            mus[i] = o_mu
            sigmas[i] = o_sigma
            rhos[i] = o_rho

            prev_x = np.zeros((1, 1, 2*args.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            prev_state = next_state_c, next_state_h

        return chunks, mus, sigmas, rhos
