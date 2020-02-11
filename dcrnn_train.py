from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import pandas as pd
import glob
import numpy as np
import random
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

from lib.utils import load_graph_data
from lib.utils import generate_seq2seq_data
from lib.utils import train_val_test_split
from lib.utils import StandardScaler
from lib.utils import data_prep

from model.dcrnn_supervisor import DCRNNSupervisor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

restart = False

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
       
        SEED = 1234  
        random.seed(SEED)  
        np.random.seed(SEED)  
        tf.set_random_seed(SEED)  
 
        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        bytes_in_use = BytesInUse()
        maxbytes_in_use = tf.contrib.memory_stats.MaxBytesInUse()
        with tf.Session(config=tf_config) as sess:
  
            # Train
            data_tag = supervisor_config.get('base_dir')
            if restart:
                data_tag = supervisor_config.get('base_dir')
                yaml_files = glob.glob('%s/*/*.yaml'%data_tag, recursive=True)
                yaml_files.sort(key=os.path.getmtime)
                config_filename = yaml_files[-1] #'config_%d.yaml' % config_id
                print(config_filename)
                with open(config_filename) as f:
                    config = yaml.load(f)
                supervisor = DCRNNSupervisor(config)
                supervisor.load(sess, config['train']['model_filename'])
            else:
                supervisor = DCRNNSupervisor(supervisor_config)
                folder = data_tag
                if not os.path.exists(folder):
                    os.makedirs(folder)
            supervisor.train(sess=sess)

             
            # Test
            supervisor = DCRNNSupervisor(supervisor_config)
            data_tag = supervisor_config.get('base_dir')
            yaml_files = glob.glob('%s/*/*.yaml'%data_tag, recursive=True)
            yaml_files.sort(key=os.path.getmtime)
            config_filename = yaml_files[-1] #'config_%d.yaml' % config_id
           
            with open(config_filename) as f:
                config = yaml.load(f)
            # Load model and evaluate
            supervisor.load(sess, config['train']['model_filename'])
            y_preds = supervisor.evaluate(sess)
            
            
            folder = 'data/results/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            df_sp = pd.DataFrame(y_preds)
            filename = os.path.join(folder, 'dcrnn_speed_bay.h5')
            df_sp.to_hdf(filename, 'results')
            
             
            
          


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
