import main
import parameters

def loop():
    while True:
        choice = int(input('predefined (1) or custom(2)? ').strip())
        if choice == 1:
            predefined()
        else:
            pass

def predefined():
    list_of_predef = ['parity', 'symmetry', 'bit_counter', 'segment_counter', 'mnist', 'wine', 'glass', 'yeast', 'seeds']
    for i in range(len(list_of_predef)):
        print(str(i) + ': ' + list_of_predef[i])
    choice = list_of_predef[int(input('Choose a set: ').strip())]

    print('\n'*5 + 'Running case', choice, 'with parameters', '\n')
    params = parameters.good_parameters(choice)
    for key, value in params.items():
        print(key+':', value)

    main.main(**params)

    print('\n'*5)

if __name__ == '__main__':
    loop()