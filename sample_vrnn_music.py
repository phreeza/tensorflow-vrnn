import tensorflow as tf

import os
import cPickle
from model_vrnn import VRNN
import numpy as np
import util

from train_vrnn import next_batch


with open(os.path.join('save-vrnn-music', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

data , means, stds = util.load_augment_data(
    util.loadf(os.path.join(saved_args.mp3_path,'Kimiko Ishizaka - J.S. Bach- -Open- Goldberg Variations, BWV 988 (Piano) - 01 Aria.mp3')),saved_args.chunk_samples)

model = VRNN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save-vrnn-music')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)
sample_data,mus,sigmas,rhos = model.sample(sess,saved_args,44100*60/saved_args.chunk_samples,start=data[:890,:],T=1.)
#sample_data[:,:saved_args.chunk_samples] = (sample_data[:,:saved_args.chunk_samples]*stds+means)
sample_data[:,:saved_args.chunk_samples] = (mus*stds+means)
sample_trace = util.write_data(np.minimum(sample_data,1.1), fname = "out-vrnn.wav")
