import tensorflow as tf

import os
import cPickle
from model_vrnn import VRNN
import numpy as np

from train_vrnn import next_batch

with open(os.path.join('save-vrnn', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = VRNN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save-vrnn')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)
sample_data,mus,sigmas = model.sample(sess,saved_args)
