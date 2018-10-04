import main
import mnist.mnist_basics as mb
import tflowtools as TFT
import tensorflow as tf
import net_config


dataset_generators = {
    'parity': TFT.gen_all_parity_cases
    , 'symmetry': TFT.gen_symvect_dataset
    , 'bit_counter': TFT.gen_vector_count_cases
    , 'segment_counter': TFT.gen_segmented_vector_cases
    , 'mnist': mb.load_all_flat_cases_ML
    , 'yeast': main.get_irvine_cases
    , 'glass': main.get_irvine_cases
    , 'wine': main.get_irvine_cases
    , 'seeds': main.get_irvine_cases
}

def good_parameters(dataset):
    return net_config.param_dict[dataset]

def get_config():
    return net_config.param_dict
