import numpy as np
import tensorflow as tf

import argparse
import glob
import time
from datetime import datetime
import os
import cPickle

import util
from model_vrnn import VRNN

from matplotlib import pyplot as plt

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict function
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
    - implement separate MDCT training and sampling version
'''

def next_batch(data, args):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in xrange(args.batch_size):
        idx = np.random.randint(1000, data.shape[0]-args.seq_length-2)
        x_batch.append(np.copy(data[idx:idx+args.seq_length]))
        y_batch.append(np.copy(data[idx+1:idx+args.seq_length+1]))
    return np.array(x_batch), np.array(y_batch)


def train(args, model):
    fnames = glob.glob(os.path.join(args.mp3_path,"*.mp3"))
    traces = np.hstack([util.loadf(fname) for fname in fnames])
    dirname = 'save-vrnn-music'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname)
    n_batches = traces.shape[0]/(args.batch_size*args.chunk_samples)
    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        if args.check_numerics:
           check = tf.add_check_numerics_ops()
        merged = tf.merge_all_summaries()
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Loaded model"

        state = model.initial_state_c, model.initial_state_h
        start = time.time()
        sess.run(tf.assign(model.lr, args.learning_rate))
        for e in xrange(args.num_epochs):
            if e%10 == 0:
                data, _, _ = util.load_augment_data(traces,args.chunk_samples)
                print "Refreshed data"
            for b in xrange(n_batches):
                x,y = next_batch(data,args)
                feed = {model.input_data: x, model.target_data: y}
                if args.check_numerics:
                    train_loss, _, _, summary, kl_loss, likelihood_loss, mu, sigma, rho, sigma_full = sess.run(
                            [model.cost, model.train_op, check, merged, model.kl_loss,
                             model.likelihood_loss, model.mu_avg, model.sigma_avg, model.rho_avg, model.sigma],
                                                                 feed)
                else:
                    train_loss, _, summary, kl_loss, likelihood_loss, mu, sigma, rho, sigma_full = sess.run(
                            [model.cost, model.train_op, merged, model.kl_loss,
                             model.likelihood_loss, model.mu_avg, model.sigma_avg, model.rho_avg, model.sigma],
                                                                 feed)
                summary_writer.add_summary(summary, e * n_batches + b)
                if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                    print "model saved to {}".format(checkpoint_path)
                end = time.time()
                print "{}/{} (epoch {}), train_loss(k+l) = {:.6f}({:.3f}+{:.3f}), time/batch = {:.1f}, std = {:.3f}, mu = {:.3f}, rho = {:.3f}" \
                    .format(e * n_batches + b,
                            args.num_epochs * n_batches,
                            e, train_loss,kl_loss,likelihood_loss, end - start, sigma, mu, rho)
                print sigma,sigma_full.mean()
                start = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=300,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=300,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--chunk_samples', type=int, default=1,
                        help='number of samples per mdct chunk')
    parser.add_argument('--check_numerics', type=bool, default=False,
                        help='run numerics checks on all nodes')
    parser.add_argument('--mp3_path', type=str, default="mp3",
                        help='run numerics checks on all nodes')
    args = parser.parse_args()

    model = VRNN(args)

    train(args, model)
