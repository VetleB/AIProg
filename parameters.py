import main
import mnist.mnist_basics as mb
import tflowtools as TFT
import tensorflow as tf

def good_parameters(dataset):
    param_dict = {
        'yeast': {
              'dims': [8, 128, 64, 32, 10]
            , 'data_source': (main.get_all_irvine_cases, {'case': 'yeast'})
            , 'steps': 50000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'HAF': tf.nn.tanh
            , 'OAF': tf.nn.sigmoid
            , 'IWR': (-.1, .1)
            , 'eint': 100
            , 'l_rate': 0.1
            , 'map_batch_size': 0
            , 'map_layers': []
        }
        , 'glass': {
              'dims': [9, 128, 64, 32, 7]
            , 'data_source': (main.get_all_irvine_cases, {'case': 'glass'})
            , 'steps': 50000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'HAF': tf.nn.tanh
            , 'OAF': tf.nn.tanh
            , 'IWR': (-1, 1)
            , 'eint': 100
            , 'l_rate': 0.1
            , 'map_batch_size': 0
            , 'map_layers': [0, 1, 2]
        }
        , 'wine': {
              'dims': [11, 128, 64, 32, 8, 6]
            , 'data_source': (main.get_all_irvine_cases, {'case': 'wine'})
            , 'steps': 150000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'HAF': tf.tanh
            , 'OAF': tf.tanh
            , 'IWR': (-.1, .1)
            , 'eint': 100
            , 'l_rate': 0.1
            , 'map_batch_size': 0
            , 'map_layers': [0, 1, 2]
        }
        , 'mnist': {
              'dims': [784, 1024, 128, 32, 10]
            , 'data_source': (mb.load_all_flat_cases_ML, {'unify': True, 'one_hot': True})
            , 'steps': 20000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'HAF': tf.tanh
            , 'OAF': tf.tanh
            , 'IWR': (-.1, .1)
            , 'eint': 100
            , 'l_rate': 0.1
            , 'case_fraction': 0.1
        }
        , 'parity': {
            'dims': [10, 50, 2]
            , 'data_source': (TFT.gen_all_parity_cases, {'num_bits': 10, 'double': True})
            , 'steps': 15000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.sigmoid
            , 'eint': 100
            , 'l_rate': 0.1
            , 'map_batch_size': 0
            , 'map_layers': []
        }
        , 'symmetry': {
            'dims': [101, 64, 2]
            , 'data_source': (TFT.gen_symvect_dataset, {'vlen': 101, 'count': 2000})
            , 'steps': 20000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'OAF': tf.nn.relu
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 0
            , 'map_layers': []
        }
        , 'bit_counter': {
            'dims': [15, 20, 16]
            , 'data_source': (TFT.gen_vector_count_cases, {'num': 500, 'size': 15})
            , 'steps': 3000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 10
            , 'map_layers': [0, 1, 2]
            , 'map_dendrograms': [1, 2]
            , 'display_weights': [1, 2]
            , 'display_biases': [1, 2]
        }
        , 'seg_counter': {
            'dims': [25, 20, 9]
            , 'data_source': (TFT.gen_segmented_vector_cases, {'vectorlen': 25, 'count': 1000, 'minsegs': 0, 'maxsegs': 8})
            , 'steps': 10000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 0
            , 'map_layers': []
        }
        , 'auto': {
            'dims': [8, 3, 8]
            , 'data_source': (TFT.gen_all_one_hot_cases, {'len': 8})
            , 'steps': 10000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'eint': 100
            , 'l_rate': 0.1
            , 'vfrac': 0
            , 'tfrac': 0
            , 'minibatch_size': 1
            , 'map_batch_size': 10
            , 'map_layers': [0, 1, 2]
            , 'map_dendrograms': [1, 2]
            , 'display_weights': [1, 2]
            , 'display_biases': [1, 2]
        }
    }
    return param_dict[dataset]
