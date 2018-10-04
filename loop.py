import main
import parameters
import tensorflow as tf


def loop():
    # Main loop
    while True:
        choice = int(input('predefined (1) or customized (2)? ').strip())
        if choice == 1:
            params, choice = get_set()
        else:
            params, choice = edit_predefined()

        print('\n' * 5 + 'Running case', choice, 'with parameters', '\n')

        print_params(params)
        print('\n'*3)

        main.main(**params)

        print('\n' * 5)


def edit_predefined():
    string_params = ['optimizer', 'loss_func']
    activation_funcs = ['HAF', 'OAF']
    int_params = ['steps', 'vint', 'minibatch_size', 'map_batch_size']
    float_params = ['l_rate', 'case_fraction', 'vfrac', 'tfrac']
    list_params = ['dims', 'IWR', 'map_layers', 'map_dendrograms', 'display_weights', 'display_biases']

    param = 'reset'
    while param == 'reset':

        params, choice = get_set(default=True)

        print_params(params)
        param = input('\n'*2 + 'Choose a parameter to edit: ').strip()
        while param != 'q' and param != 'reset':
            if param in string_params:
                new_param = get_string_param(param)
            elif param in activation_funcs:
                new_param = get_activation_func()
            elif param in int_params:
                new_param = int(input(param + ' value: ').strip())
            elif param in float_params:
                new_param = float(input(param + ' value: ').strip())
            elif param in list_params:
                new_param = get_list_param(param)
            elif param == 'data_source':
                print('\n' + 'Please change in net_config')
                param = input('\n' * 2 + 'Choose a parameter to edit: ').strip()
                continue
            else:
                print('\n' + 'Not a valid parameter')
                param = input('\n' * 2 + 'Choose a parameter to edit: ').strip()
                continue

            params[param] = new_param

            print_params(params)
            param = input('\n'*2 + 'Choose a parameter to edit: ').strip()


    return params, choice


# WIP, not finished for project 1
def get_data_source(params):
    data_param = input('\n' + 'generator (1) or kwargs (2)? ').strip()

    generator = ''
    kwargs = params['data_source'][1]

    while data_param != 'q':
        data_param == int(data_param)
        if data_param == 1:
            print('\n' + 'Your options are: ')
            for key, value in parameters.dataset_generators.items():
                print(key, value)

            new_param = input('Your choice: ')

            if new_param not in parameters.dataset_generators.keys():
                print('\n' + 'Invalid generator')
                data_param = input('\n' + 'generator (1) or kwargs (2)? ').strip()
                continue

            generator = new_param

        elif data_param == 2:
            print('\n' + 'Add or delete kwarg?')

        data_param = input('\n' + 'generator (1) or kwargs (2)? ').strip()


# Read input and turn it into a list
def get_list_param(param):
    values = str(input('\n'*2 + 'Values for list ' + param + ' (separator=,): ').strip())
    string_list = values.split(',')
    if param == 'IWR':
        new_param = (float(string_list[0]), float(string_list[1]))
    else:
        new_param = [int(s) for s in string_list]
    return new_param


# List options for various string-type parameters
def get_string_param(param):
    opts = ['gd', 'rms', 'adam', 'adagrad']
    losses = ['mse', 'x_entropy']
    new_param = ''
    if param == 'optimizer':
        while new_param not in opts:
            print('\n' + 'Your options for ' + param + ' are:')
            for opt in opts:
                print('\t' + opt)
            new_param = input('Your choice: ')
    elif param == 'loss_func':
        while new_param not in losses:
            print('\n' + 'Your options for ' + param + ' are:')
            for loss in losses:
                print('\t' + loss)
            new_param = input('Your choice: ')

    return new_param


def get_activation_func():
    act_funcs = {
        'tanh': tf.tanh,
        'sigmoid': tf.sigmoid,
        'relu': tf.nn.relu,
        'softmax': tf.nn.softmax
    }

    funcs = act_funcs.keys()
    new_param = ''

    while new_param not in funcs:
        print('Your options are:')
        for f in funcs:
            print('\t' + f)
        new_param = input('Your choice: ')

    return act_funcs[new_param]


# Choose a set of parameters either for running or for editing further
def get_set(default=False):
    list_of_predef = list(parameters.get_config().keys())
    if default:
        list_of_predef.append('custom')

    for i in range(len(list_of_predef)):
        print(str(i) + ': ' + list_of_predef[i])

    choice = list_of_predef[int(input('\n' + 'Choose a set: ').strip())]

    params = parameters.good_parameters(choice)

    return params, choice


def print_params(params):
    print('\n')
    for key, value in params.items():
        print(key+':', value)


if __name__ == '__main__':
    loop()
