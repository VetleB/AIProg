import main
import mnist.mnist_basics as mb
import tflowtools as TFT
import tensorflow as tf

def good_parameters(dataset):
    param_dict = {
        'yeast': {
              'dims': [8, 20, 10]
            , 'data_source': (main.get_all_irvine_cases, {'case': 'yeast'})
            , 'steps': 2000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 15
            , 'map_layers': [0, 1, 2]
        }
        , 'glass': {
              'dims': [9, 20, 7]
            , 'data_source': (main.get_all_irvine_cases, {'case': 'glass'})
            , 'steps': 20000
            , 'optimizer': 'gd'
            , 'loss_func': 'mse'
            , 'OAF': tf.nn.softmax
            , 'HAF': tf.nn.sigmoid
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 0
            , 'map_layers': [0, 1, 2]
        }
        , 'wine': {
              'dims': [11, 20, 6]
            , 'data_source': (main.get_all_irvine_cases, {'case': 'wine'})
            , 'steps': 20000
            , 'optimizer': 'adam'
            , 'loss_func': 'x_entropy'
            , 'OAF': tf.sigmoid
            , 'HAF': tf.sigmoid
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 10
            , 'map_layers': [0, 1, 2]
        }
        , 'mnist': {
              'dims': [784, 100, 20, 10]
            , 'data_source': (mb.load_all_flat_cases_ML, {'unify': True, 'one_hot': True})
            , 'steps': 5000
            , 'optimizer': 'gd'
            , 'loss_func': 'x_entropy'
            , 'eint': 100
            , 'l_rate': 1
            , 'case_fraction': 0.1
        }
        , 'parity': {
            'dims': [10, 20, 2]
            , 'data_source': (TFT.gen_all_parity_cases, {'num_bits': 10, 'double': True})
            , 'steps': 5000
            , 'optimizer': 'gd'
            , 'loss_func': 'mse'
            , 'eint': 100
            , 'l_rate': 1
            , 'map_batch_size': 0
            , 'map_layers': [0, 1]
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
            , 'map_batch_size': 15
            , 'map_layers': [0, 1, 2]
        }
    }
    return param_dict[dataset]
