from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import pandas as pd
import glob
import numpy as np

from input_files.utils import load_graph_data
from input_files.utils import generate_seq2seq_data
from input_files.utils import train_val_test_split
from input_files.utils import StandardScaler
from traffic_tfrecords import write_as_tfrecords
from traffic_tfrecords import _parse_record_fn

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        # Data preprocessing 
        traffic_df_filename = supervisor_config['data']['hdf_filename'] 
        df_data = pd.read_hdf(traffic_df_filename)
        sensor_ids = list(df_data.columns)
        supervisor_config['model']['num_nodes'] = num_nodes = len(sensor_ids)


        validation_ratio = supervisor_config.get('data').get('validation_ratio')
        test_ratio = supervisor_config.get('data').get('test_ratio')
        df_train, df_val, df_test = train_val_test_split(df_data, val_ratio=validation_ratio, test_ratio=test_ratio)

        batch_size = supervisor_config.get('data').get('batch_size')
        val_batch_size = supervisor_config.get('data').get('val_batch_size')
        test_batch_size = supervisor_config.get('data').get('test_batch_size')
        horizon = supervisor_config.get('model').get('horizon')
        seq_len = supervisor_config.get('model').get('seq_len')
        scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
        
        # In case of multiple partitions 
        sensor_ids_filesname = supervisor_config.get('data').get('sensor_ids')
        loop_df = pd.read_csv(sensor_ids_filesname)
        loop_ids = loop_df['sensor_id'].astype('str').tolist()
        graph_partition_filesname = supervisor_config.get('data').get('graph_partitions')
        partition = np.genfromtxt(graph_partition_filesname, dtype=int, delimiter="\n", unpack=False)
        max_node =  max(np.bincount(partition))

        folder = 'TFrecords/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        partition_ids = np.unique(partition)
        for p in partition_ids: 
        
            indices = partition==p
            part_df = loop_df[indices]
            loop_ids = part_df['sensor_id'].astype('str').tolist()
            h5files = df_data[loop_ids] 
            num_nodes = len(loop_ids)
            if num_nodes < max_node:
                pad = max_node - num_nodes
                solution = (['%i' %i for i in range(pad)])
                d = dict.fromkeys(solution, 0)
                h5files = h5files.assign(**d)
            df_train, df_val, df_test = train_val_test_split(h5files, val_ratio=validation_ratio, test_ratio=test_ratio)
            
            
            x_train, y_train = generate_seq2seq_data(df_train, batch_size, seq_len, horizon, max_node, 'train', scaler)
            x_val, y_val = generate_seq2seq_data(df_val, val_batch_size, seq_len, horizon, max_node, 'val', scaler)
            x_test, y_test = generate_seq2seq_data(df_test, test_batch_size, seq_len, horizon, max_node, 'test', scaler)
            
            # Write TFrecords file
            write_as_tfrecords(x_train, y_train, 'train_' + str(p), './%s'%folder, num_shards=1)
            write_as_tfrecords(x_val, y_val , 'val_'+ str(p), './%s'%folder)
            write_as_tfrecords(x_test, y_test , 'test_' + str(p), './%s'%folder)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)




