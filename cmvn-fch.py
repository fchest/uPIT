#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Sining Sun (Northwestern Polytechnical University, China)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
sys.path.append('.')

from io_funcs.signal_processing import audiowrite, stft,istft 

import io_funcs.kaldi_io as kio
# from model.blstm import LSTM
from io_funcs.tfrecords_io import get_padded_batch
from local.utils import pp, show_all_variables

FLAGS = None

class LSTM(object):
    def __init__(self, inputs, lengths):
        self._inputs = inputs
        self._lengths = lengths

def read_list_file(name, batch_size):
    file_name = os.path.join(FLAGS.lists_dir, name + ".lst")
    if not os.path.exists(file_name):
        tf.logging.fatal("File doesn't exist %s", file_name)
        sys.exit(-1)
    config_file = open(file_name)
    tfrecords_lst = []
    for line in config_file:
        utt_id = line.strip().split()[0]
        tfrecords_name = utt_id
        if not os.path.exists(tfrecords_name):
            tf.logging.fatal("TFRecords doesn't exist %s", tfrecords_name)
            sys.exit(-1)
        tfrecords_lst.append(tfrecords_name)
    num_batches = int(len(tfrecords_lst) / batch_size + 0.5)
    return tfrecords_lst, num_batches


def train_one_epoch(sess, coord, tr_model, tr_num_batches):
    """Runs the model one epoch on given data."""
    tr_mix = sess.run(tr_model._inputs)
    tr_mix1 = tr_mix
    for batch in xrange(1, tr_num_batches):
        if coord.should_stop():
            break
        # tr_mix, lengths = sess.run(tr_model._inputs, tr_model._lengths)
        tr_mix = sess.run(tr_model._inputs)
        tr_mix1 = np.concatenate((tr_mix1, tr_mix), axis=0)
    print("the shape of tr_mix1: ")
    print(np.shape(tr_mix1))
    X_mean = tr_mix1.mean(axis=0)
    X_std = tr_mix1.std(axis=0)
    return X_mean, X_std


def train():
    tr_tfrecords_lst, tr_num_batches = read_list_file("tr_tf", FLAGS.batch_size)
  
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                tr_mixed,tr_labels,tr_genders,tr_lengths = get_padded_batch(
                    tr_tfrecords_lst, FLAGS.batch_size, FLAGS.input_size*2,
                    FLAGS.output_size*2, num_enqueuing_threads=FLAGS.num_threads,
                    num_epochs=FLAGS.max_epochs)

                tr_inputs = tf.slice(tr_mixed, [0,0,0], [-1,-1, FLAGS.input_size])
                tr_inputs = tf.reshape(tr_inputs, [-1, FLAGS.input_size])
     

        with tf.name_scope('model'):
            tr_model = LSTM(tr_inputs, tr_lengths)
            # tr_model = LSTM(FLAGS, tr_inputs, tr_labels,tr_lengths,tr_genders)
            # tr_model and val_model should share variables
            tf.get_variable_scope().reuse_variables
        show_all_variables()
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
            # Prevent exhausting all the gpu memories.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        #sess = tf.InteractiveSession(config=config)
        sess = tf.Session(config=config)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)	
        try:
            # Cross validation before training.
            # lengths = sess.run(tr_lengths)
            # print('lengths shape: ')
            # print(np.shape(lengths))
            # print(lengths)
            # tr_mix = sess.run(tr_labels)
            # print('tr_mix shape: ')
            # print(np.shape(tr_mix))
            X_mean, X_std = train_one_epoch(sess, coord, tr_model, tr_num_batches)
            # print(np.shape(X_mean))
            np.savetxt('tr_mean_good1',X_mean,fmt="%s")
            np.savetxt('tr_std_good1',X_std,fmt="%s")

                
        except Exception, e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            # Terminate as usual.  It is innocuous to request stop twice.
            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)

        tf.logging.info("Done training")
        sess.close()

def main(_):
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    train()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--decode',
        type=int,
        default=0,
        #action='store_true',
        help="Flag indicating decoding or training."
    )
    parser.add_argument(
        '--resume_training',
        type=str,
        default='False',
        help="Flag indicating whether to resume training from cptk."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/wsj0',
        help="Directory of train, val and test data."
    )
    parser.add_argument(
        '--lists_dir',
        type=str,
        default='lists',
        help="Directory to load train, val and test data."
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=129,
        help="The dimension of input."
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=129,
        help="The dimension of output."
    )
    parser.add_argument(
        '--czt_dim',
        type=int,
        default=0,
        help="chrip-z transform feats dimension. it should be 0 if you just use fft spectrum feats"
    )
 
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=128,
        help="Number of rnn units to use."
    )
    parser.add_argument(
        '--rnn_num_layers',
        type=int,
        default=2,
        help="Number of layer of rnn model."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Mini-batch size."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help="Initial learning rate."
    )
    parser.add_argument(
        '--min_epochs',
        type=int,
        default=30,
        help="Min number of epochs to run trainer without halving."
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help="Max number of epochs to run trainer totally."
    )
    parser.add_argument(
        '--halving_factor',
        type=float,
        default=0.5,
        help="Factor for halving."
    )
    parser.add_argument(
        '--start_halving_impr',
        type=float,
        default=0.003,
        help="Halving when ralative loss is lower than start_halving_impr."
    )
    parser.add_argument(
        '--end_halving_impr',
        type=float,
        default=0.01,
        help="Stop when relative loss is lower than end_halving_impr."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=12,
        help='The num of threads to read tfrecords files.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='exp/deepcluster_test',
        help="Directory to put the train result."
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.8,
        help="Keep probability for training dropout."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.0,
        help="The max gradient normalization."
    )
    parser.add_argument(
        '--assign',
        type=str,
        default='def',
        help="Assignment method, def or opt"
    )
    parser.add_argument(
        '--model_type',
        type=str, default='LSTM',
        help="BLSTM or LSTM"
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp.pprint(FLAGS.__dict__)
    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
