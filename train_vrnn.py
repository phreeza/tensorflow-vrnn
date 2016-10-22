import numpy as np
import tensorflow as tf

import argparse
import glob
import time
from datetime import datetime
import os
import cPickle

from model_vrnn import VRNN

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=800,
                     help='size of RNN hidden state')
  parser.add_argument('--latent_size', type=int, default=800,
                     help='size of latent space')
  parser.add_argument('--batch_size', type=int, default=25,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=100,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=500,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
  parser.add_argument('--chunk_samples', type=int, default=1024,
                     help='number of samples per mdct chunk')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  args = parser.parse_args()
  train(args)

def train(args):
    dirname = 'save-vrnn'
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
      cPickle.dump(args, f)

    model = VRNN(args)
    # load previously trained model if applicable
    ckpt = tf.train.get_checkpoint_state(dirname)
    if ckpt:
      model.load_model(dirname)

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('logs/'+datetime.now().isoformat().replace(':','-'), sess.graph)
        check = tf.add_check_numerics_ops()
        merged = tf.merge_all_summaries()
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        start = time.time()
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = model.initial_state
            for b in xrange(100):
                t0 = np.random.randn(args.batch_size,1,(2*args.chunk_samples))
                x = np.sin(2*np.pi*(np.arange(args.seq_length)[np.newaxis,:,np.newaxis]/30.+t0)) + np.random.randn(args.batch_size,args.seq_length,(2*args.chunk_samples))*0.1
                y = np.sin(2*np.pi*(np.arange(1,args.seq_length+1)[np.newaxis,:,np.newaxis]/30.+t0)) + np.random.randn(args.batch_size,args.seq_length,(2*args.chunk_samples))*0.1
                y[:,:,args.chunk_samples:] = 0.
                x[:,:,args.chunk_samples:] = 0.
                feed = {model.input_data: x, model.target_data: y}
                train_loss, _, cr, summary, sigma = sess.run([model.cost, model.train_op, check, merged, model.sigma], feed)
                summary_writer.add_summary(summary, e * 100 + b)
                if (e * 100 + b) % args.save_every == 0 and ((e * 100 + b) > 0):
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * 100 + b)
                    print "model saved to {}".format(checkpoint_path)
                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}/{:.3f}" \
                    .format(e * 100 + b,
                            args.num_epochs * 100,
                            e, args.chunk_samples*train_loss, end - start, (sigma[:,200:]).mean(axis=0).mean(axis=0),(sigma[:,:200]).mean(axis=0).mean(axis=0))
                start = time.time()

if __name__ == '__main__':
  main()


