from config import SAVE_DIR
from utils import create_dir, pickle_save
from model_vrnn import VRNN
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import pickle

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict function
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
    - implement separate MDCT training and sampling version
'''

def next_batch(args):
    t_offset = np.random.randn(args.batch_size, 1, (2 * args.chunk_samples))
    mixed_noise = np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    x = np.random.randn(args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
        + mixed_noise*0.1 
        + np.sin(2 * np.pi * (np.arange(args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t_offset)) 

    y = np.random.randn(args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 
        + mixed_noise*0.1
        + np.sin(2 * np.pi * (np.arange(1, args.seq_length+1)[np.newaxis, :, np.newaxis] / 10. + t0)) 
    y[:, :, args.chunk_samples:] = 0.
    x[:, :, args.chunk_samples:] = 0.
    return x, y

def train(args, model):
    create_dir(SAVE_DIR)
    pickle_path = os.path.join(SAVE_DIR, 'config.pkl')
    pickle_save(args, pickle_path)

    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
    n_batches = 100
    with tf.Session() as sess:0
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Loaded model")

        for epoch in range(args.num_epochs):
            # Learning rate decay
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** epoch)))
            
            for b in range(n_batches):
                x, y = next_batch(args)
                feed_dict = {model.input_data: x, model.target_data: y}
                train_loss, _, cr, sigma= sess.run([model.cost, model.train_op, check, model.sigma], feed_dict = feed_dict)

                if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                print("{}/{}(epoch {}), train_loss = {:.6f}, std = {:.3f}".format(e * n_batches + b, args.num_epochs * n_batches, e, args.chunk_samples * train_loss, sigma.mean(axis=0).mean(axis=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=3,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=3,
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
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--chunk_samples', type=int, default=1,
                        help='number of samples per mdct chunk')
    args = parser.parse_args()
    model = VRNN(args)
    train(args, model)
