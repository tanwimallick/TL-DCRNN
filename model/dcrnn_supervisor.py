from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml
import pandas as pd
import random

from lib.utils import load_graph_data
from lib import utils, metrics
from lib.metrics import masked_mae_loss
from lib.utils import train_val_test_split
from lib.utils import StandardScaler
from lib.utils import data_prep
from lib.utils import generate_seq2seq_data


from model.dcrnn_model import DCRNNModel
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, kwargs):
        #nvidia_smi.nvmlInit()
        #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data preparation
        self.batch_size = self._data_kwargs.get('batch_size')
        self.val_batch_size = self._data_kwargs['val_batch_size']
        self.test_batch_size = 1
        self.horizon = self._model_kwargs.get('horizon')
        self.seq_len = self._model_kwargs.get('seq_len')

        self.validation_ratio = self._data_kwargs['validation_ratio']
        self.test_ratio = self._data_kwargs['test_ratio']

        sensor_filename = self._data_kwargs['sensor_filename']
        self.sensor_df = pd.read_csv(sensor_filename)
        distance_filename = self._data_kwargs['distance_filename']
        self.dist_df = pd.read_csv(distance_filename)
        partition_filename = self._data_kwargs['partition_filename']
        self.partition = np.genfromtxt(partition_filename, dtype=int, delimiter="\n", unpack=False)
        self.clusters = np.unique(self.partition) 
        self.max_node = max(np.bincount(self.partition)) 
        self._model_kwargs['num_nodes'] = self.max_node


        # Build models.
        #scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(is_training=True, 
                                               batch_size=self._data_kwargs['batch_size'],
                                               **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(is_training=False, 
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr, )

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        self.preds_test = self._test_model.outputs
        self.labels_test = self._test_model.labels[..., :output_dim]

        null_val = 0.
        self._loss_fn = masked_mae_loss(null_val)
        # self._loss_fn = masked_mae_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)
        #print('output labels', labels.shape)
        self._test_loss = self._loss_fn(preds=self.preds_test, labels=self.labels_test)


        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, adj_mx, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        outputs = []
        if training:
             fetches = {
                 'loss': self._train_loss,
                 'mae': self._train_loss,
                 'global_step': tf.train.get_or_create_global_step()
             }
        else:
            fetches = {
                 'loss': self._test_loss,
                 'mae': self._test_loss,
                 'global_step': tf.train.get_or_create_global_step()
            }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        while True:
            try:
                x, y = sess.run(data_generator)
                feed_dict = {
                    model.inputs: x,
                    model.labels: y,
                    model.adj_mx: adj_mx 
                }
                vals = sess.run(fetches, feed_dict=feed_dict)

                losses.append(vals['loss'])
                maes.append(vals['mae'])
                if writer is not None and 'merged' in vals:
                    writer.add_summary(vals['merged'], global_step=vals['global_step'])
                if return_output:
                    outputs.append(vals['outputs'])

            except tf.errors.OutOfRangeError:
                break

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })


    def get_adjacency_matrix(self, distance_df, sensor_ids, normalized_k=0.1):
        """
        :param distance_df: data frame with three columns: [from, to, distance].
        :param sensor_ids: list of sensor ids.
        :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
        """
        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf
    
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i
    
        # Fills cells in the matrix with distances.
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        
        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < normalized_k] = 0
        
        if num_sensors < self.max_node:
            pad = self.max_node - num_sensors
            adj_mx = np.append(adj_mx, np.zeros([len(adj_mx),pad]),1) # pad column
            adj_mx = np.append(adj_mx, np.zeros([pad, adj_mx.shape[1]]),0) # pad row
            np.fill_diagonal(adj_mx, 1)

        return adj_mx

    @staticmethod
    def _parse_record_fn(data_record):
        features = {
            'x_shape': tf.FixedLenFeature(
                [], dtype=tf.string, default_value=''),
            'y_shape': tf.FixedLenFeature(
                [], dtype=tf.string, default_value=''),
            'x': tf.FixedLenFeature(
                [], dtype=tf.string, default_value=''),
            'y': tf.FixedLenFeature(
                [], dtype=tf.string, default_value=''),
        }
        sample = tf.parse_single_example(data_record, features)

        x_shape = tf.decode_raw(sample['x_shape'], tf.int64)
        y_shape = tf.decode_raw(sample['y_shape'], tf.int64)
        x = tf.decode_raw(sample['x'], tf.float64)
        y = tf.decode_raw(sample['y'], tf.float64)
        x = tf.reshape(x, shape=x_shape)
        y = tf.reshape(y, shape=y_shape)
        return x, y
        
    
    def cluster_data(self, cluster):
        indices = self.partition==cluster
        part_df = self.sensor_df[indices]
        distance_df = self.dist_df.loc[(self.dist_df['from'].isin(part_df['sensor_id'])) & (self.dist_df['to'].isin(part_df['sensor_id']))]
        distance_df = distance_df.reset_index(drop=True)
        distance_df['from'] = distance_df['from'].astype('str')
        distance_df['to'] = distance_df['to'].astype('str')
        sensor_ids = part_df['sensor_id'].astype(str).values.tolist()

        cdata = self._data[sensor_ids] # data required 
        if num_nodes < self.max_node:
            pad = self.max_node - num_nodes
            solution = (['%i' %i for i in range(pad)])
            d = dict.fromkeys(solution, 0)
            cdata = cdata.assign(**d)

        sp_train, sp_val = train_val_test_split(cdata, val_ratio=self.validation_ratio, test_ratio=self.test_ratio)
        sp_scaler = StandardScaler(mean=sp_train.values.mean(), std=sp_train.values.std())
        train_x, train_y = data_prep(df_in=[sp_train], df_out=[sp_train], 
                        in_scaler=[sp_scaler], out_scaler=[sp_scaler], num_nodes=self.max_node)
        data_train = generate_seq2seq_data(train_x, train_y, sp_train.shape[0], self.batch_size, self.seq_len, self.horizon, self.max_node, 'train')
        
        val_x, val_y = data_prep([sp_val], [sp_val], 
                    in_scaler=[sp_scaler], out_scaler=[sp_scaler], num_nodes=self.max_node)
        data_val = generate_seq2seq_data(val_x, val_y, sp_val.shape[0], self.val_batch_size, self.seq_len, self.horizon, self.max_node, 'val')
        data_train.update(data_val)

        node_count = len(sensor_ids)       
        adj_mx = self.get_adjacency_matrix(distance_df, sensor_ids)
 
        #return data_train, node_count, adj_mx
        return node_count, adj_mx

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        #L = tf.SparseTensor(indices, L.data, L.shape)
        #return tf.sparse_reorder(L)
        return tf.SparseTensorValue(indices, L.data, L.shape)

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
            #print('validation loss', val_loss)
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training ...')


        #cluster_arr = [38,0,62, 56, 52, 20, 48, 28, 46, 30, 33, 39, 24, 16, 14, 58]
        #remaining_list = list(set(self.clusters) - set(cluster_arr))
        #setOfSix = []
        #while len(setOfSix) < 15:
        #    setOfSix.append(random.choice(remaining_list))        
        #sclusters = cluster_arr + setOfSix

        while self._epoch <= epochs:
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            
            self.node_count_seen = 0
            self.accumulated_training_loss = 0

            # generate random number
            half_length = int(len(self.clusters)/2)
            sclusters = self.clusters[0:half_length]

            random.shuffle(sclusters)
            for cluster in sclusters:

                node_count, adj_mx = self.cluster_data(cluster)
                train_data_path = self._kwargs['data'].get('dataset_dir') + '/train_' + str(cluster) + '.tfrecords'
                train_dataset = tf.data.TFRecordDataset([train_data_path])
                train_dataset = train_dataset.map(self._parse_record_fn)
                train_iterator = train_dataset.make_one_shot_iterator()
                train_next_element = train_iterator.get_next()

                val_data_path = self._kwargs['data'].get('dataset_dir') + '/val_' + str(cluster) + '.tfrecords'
                val_dataset = tf.data.TFRecordDataset([val_data_path])
                val_dataset = val_dataset.map(self._parse_record_fn)
                val_iterator = val_dataset.make_one_shot_iterator()
                val_next_element = val_iterator.get_next()

                adj_mx = utils.calculate_random_walk_matrix(adj_mx).T
                adj_mx = self._build_sparse_matrix(adj_mx)

                train_results = self.run_epoch_generator(sess, self._train_model, train_next_element,
                                                     adj_mx,
                                                     training=True,
                                                     writer=self._writer)
                train_loss, train_mae = train_results['loss'], train_results['mae']

                if train_loss > 1e5:
                    self._logger.warning('Gradient explosion detected. Ending...')
                    break
                val_results = self.run_epoch_generator(sess, self._test_model, val_next_element, adj_mx, training=False)
                val_loss, val_mae = np.asscalar(val_results['loss']), np.asscalar(val_results['mae'])


            end_time = time.time()
            message = 'Epoch [{}/{}]  train_mae: {:.4f}, val_mae: {:.4f} lr:{:.6f} {:.1f}s'.format(
            self._epoch, epochs, train_loss, val_loss, new_lr, (end_time - start_time))
            self._logger.info(message)

            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_loss)
            # Increases epoch.
            self._epoch += 1
            sys.stdout.flush()

        return np.min(history)


    def evaluate(self, sess, **kwargs):
        
        y_preds_all = []
        half_length = int(len(self.clusters)/2) 
        sclusters =  self.clusters[0:32]
        for cluster in sclusters:
      
            node_count, adj_mx = self.cluster_data(cluster)      
            adj_mx = utils.calculate_random_walk_matrix(adj_mx).T
            adj_mx = self._build_sparse_matrix(adj_mx)
            global_step = sess.run(tf.train.get_or_create_global_step())
            scaler_path = self._kwargs['data'].get('dataset_dir') + '/scaler.npy'
            scaler_data_ = np.load(scaler_path)
            mean, var = scaler_data_[0], scaler_data_[1]
            scaler = StandardScaler(mean=mean, std=var)

            # change val to test before run
            test_data_path = self._kwargs['data'].get('dataset_dir') + '/test_' + str(cluster) + '.tfrecords'
            test_dataset = tf.data.TFRecordDataset([test_data_path])
            test_dataset = test_dataset.map(self._parse_record_fn)
            test_dataset = test_dataset.make_one_shot_iterator()
            test_next_element = test_dataset.get_next()


            test_results = self.run_epoch_generator(sess, self._test_model,
                                                test_next_element, adj_mx,
                                                return_output=True,
                                                training=False)
            test_loss, y_preds = test_results['loss'], test_results['outputs']
            utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)
    
            y_preds = np.concatenate(y_preds, axis=0)
            y_preds = scaler.inverse_transform(y_preds[:, self.horizon-1, :, 0])
            y_preds = y_preds[:,0:node_count]
            
            y_preds_all.append(y_preds)

        y_preds_all = np.concatenate(y_preds_all, axis=1)
        return y_preds_all


    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)


    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['val_loss'] = val_loss
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']
