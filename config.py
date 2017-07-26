#=================================PATH=========================#

SAVE_DIR = './save/'

#======================VRNN configuration=======================#

class VRNNConfig(object):
    def __init__(self):
        self.rnn_size = 3 # num of hidden states in RNN
        self.latent_size = 3 # size of latent space

        self.seq_length = 100 # RNN sequence length
        self.chunk_samples = 1 # number of samples per mdct chunk

        self.num_epochs = 100
        self.batch_size = 3000
        self.n_batches = 100
        self.save_every = 500

        self.grad_clip = 10 # clip gradients at this value
        self.decay_rate = 1.
        self.learning_rate = 0.0005
