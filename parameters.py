import main
import mnist.mnist_basics as mb
import tflowtools as TFT
import tensorflow as tf

def good_parameters(dataset):
    param_dict = {
        'yeast': {
              'dims': [8, 128, 64, 32, 10]
            , 'data_source': (main.get_irvine_cases, {'case': 'yeast'})
            , 'steps': 50000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 0.1
            , 'HAF': tf.tanh
            , 'OAF': tf.sigmoid
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'glass': {
              'dims': [9, 128, 64, 32, 7]
            , 'data_source': (main.get_irvine_cases, {'case': 'glass'})
            , 'steps': 40000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 0.1
            , 'HAF': tf.tanh
            , 'OAF': tf.tanh
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 128
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'wine': {
              'dims': [11, 64, 32, 16, 8, 6]
            , 'data_source': (main.get_irvine_cases, {'case': 'wine'})
            , 'steps': 75000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 0.1
            , 'HAF': tf.tanh
            , 'OAF': tf.tanh
            , 'IWR': (-.1, .1)
            , 'case_fraction': 0.1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'mnist': {
              'dims': [784, 1024, 128, 32, 10]
            , 'data_source': (mb.load_all_flat_cases_ML, {'unify': True, 'one_hot': True})
            , 'steps': 1000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 0.1
            , 'HAF': tf.tanh
            , 'OAF': tf.tanh
            , 'IWR': (-.1, .1)
            , 'case_fraction': 0.1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'seeds': {
            'dims': [7, 21, 3]
            , 'data_source': (main.get_irvine_cases, {'case': 'seeds'})
            , 'steps': 5000
            , 'optimizer': 'gd'
            , 'loss_func': 'mse'
            , 'l_rate': 0.1
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.relu
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'parity': {
            'dims': [10, 50, 2]
            , 'data_source': (TFT.gen_all_parity_cases, {'num_bits': 10, 'double': True})
            , 'steps': 15000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 0.1
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.sigmoid
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'symmetry': {
            'dims': [101, 64, 2]
            , 'data_source': (TFT.gen_symvect_dataset, {'vlen': 101, 'count': 2000})
            , 'steps': 20000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 1
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.relu
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'bit_counter': {
            'dims': [15, 20, 16]
            , 'data_source': (TFT.gen_vector_count_cases, {'num': 500, 'size': 15})
            , 'steps': 3000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 1
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.relu
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'segment_counter': {
            'dims': [25, 20, 9]
            , 'data_source': (TFT.gen_segmented_vector_cases, {'vectorlen': 25, 'count': 1000, 'minsegs': 0, 'maxsegs': 8})
            , 'steps': 10000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 1
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.relu
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
        , 'auto': {
            'dims': [8, 3, 8]
            , 'data_source': (TFT.gen_all_one_hot_cases, {'len': 8})
            , 'steps': 10000
            , 'optimizer': 'adagrad'
            , 'loss_func': 'x_entropy'
            , 'l_rate': 0.1
            , 'HAF': tf.nn.relu
            , 'OAF': tf.nn.relu
            , 'IWR': (-.1, .1)
            , 'case_fraction': 1
            , 'vint': 1000
            , 'vfrac': 0.1
            , 'tfrac': 0.1
            , 'minibatch_size': 64
            , 'map_batch_size': 0
            , 'map_layers': []
            , 'map_dendrograms': []
            , 'display_weights': []
            , 'display_biases': []
        }
    }
    return param_dict[dataset]
