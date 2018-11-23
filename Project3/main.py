import random

import play
import versus
import hex
import anet
import topp

def main():

    anet_layers = {
        3: [120, 64]
        , 4: [120, 64]
        , 5: [512, 256, 128, 64]
        , 6: [240, 128, 64]
        , 7: [240, 128, 64]
        , 8: [240, 128, 64]
    }

    ###################
    # Game parameters #
    ###################
    side_length = 5
    rollouts = (400, 'r')   # r -> amount ; s -> seconds
    player_start = -1       # -1 -> random
    verbose = False

    play_game = False
    batch_size = 8
    train_epochs = 200
    rbuf_mbs = 64
    topp_training = True
    topp_k = 5

    play_versus = False
    num_versus_matches = 1000
    pre_train = False
    pre_train_epochs = 1
    pre_train_max_amount = 2000

    run_topp = False
    games_per_series = 20

    run_good_topp = True

    ###################
    # Anet parameters #
    ###################
    lrate = 2
    optimizer = 'sgd'
    haf = 'tanh'
    oaf = 'sigmoid'
    loss = 'mean_absolute_error'
    hidden_layers = anet_layers[side_length]
    load_existing = True
    anet_name = "anet_5x5"


    #########
    # Setup #
    #########

    game = hex.Hex
    game_kwargs = {'side_length': side_length, 'verbose': verbose}

    list_of_topps = []

    input_layer_size = 2*side_length**2+2
    output_size = side_length**2
    anet_name = 'anet_' + str(side_length) + 'x' + str(side_length) if not anet_name else anet_name
    layers = [input_layer_size]
    layers.extend(hidden_layers)
    layers.append(output_size)
    anet_kwargs = {'layers': layers
        , 'haf': haf
        , 'oaf': oaf
        , 'loss': loss
        , 'optimizer': optimizer
        , 'lrate': lrate
        , 'model_name': anet_name
        , 'pre_train_epochs': pre_train_epochs
        , 'load_existing': load_existing}



    ############
    # Gameplay #
    ############

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, rbuf_mbs, anet_kwargs=anet_kwargs, train_epochs=train_epochs)

    if play_game:
        list_of_topps = p.play_game(topp=topp_training, topp_k=topp_k)

    #game_kwargs = {'side_length': side_length, 'verbose': False}

    if play_versus:
        anet_player = anet.Anet(**anet_kwargs)

        if pre_train:
            all_c = p.get_all_cases()
            pre_train_cases = all_c[0:pre_train_max_amount]
            random.shuffle(pre_train_cases)
            anet_player.train_on_cases(pre_train_cases)
            anet_player.save_model()
        v = versus.Versus(game_kwargs, game, num_versus_matches, player_start, player1=anet_player, player2=None)
        v.match()

    if run_topp:
        list_of_topps = list_of_topps if list_of_topps else ['anet_5x5_topp_0', 'anet_5x5_topp_50', 'anet_5x5_topp_100', 'anet_5x5_topp_150', 'anet_5x5_topp_200']
        topp_ = topp.Topp(list_of_topps, game_kwargs, game, games_per_series)

        # an = topp_.agents[-1]
        # v.player1 = an
        # v.match()

        topp_.run_topp()
        # topp_.display_scores()

    if run_good_topp:
        list_of_topps = ['anet_5x5_topp_0', 'anet_5x5_topp_50', 'anet_5x5_topp_100', 'anet_5x5_topp_150', 'anet_5x5_topp_200']
        topp_ = topp.Topp(list_of_topps, game_kwargs, game, games_per_series)
        topp_.run_topp()


if __name__ == '__main__':
    main()
