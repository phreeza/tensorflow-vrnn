from train import next_batch
from utils import pickle_load
from model import VRNN
from config import SAVE_DIR
import tensorflow as tf
import numpy as np
import pickle
import os

load_path = os.path.join(SAVE_DIR, 'config.pkl')
loaded_args = pickle_load(load_path)

model = VRNN(loaded_args, True)
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
print("loading model: ", ckpt.model_checkpoint_path)
saver.restore(sess, ckpt.model_checkpoint_path)
sample_data,mus,sigmas = model.sample(sess, loaded_args)