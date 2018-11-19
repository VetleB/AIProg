import play
import versus
import hex
import anet
import topp

def main():

    anet_layers = {
        3: [120, 64]
        ,4: [120, 64]
        ,5: [200, 128, 64]
    }


    ###################
    # Game parameters #
    ###################
    side_length = 4
    rollouts = (400, 'r')   # r -> amount ; s -> seconds
    player_start = -1       # -1 -> random
    verbose = False

    play_game = False
    batch_size = 100
    train_epochs = 200
    topp_training = True
    topp_k = 5

    play_versus = False
    num_versus_matches = 1000
    pre_train = False
    pre_train_epochs = 50

    run_topp = True
    games_per_series = 25

    ###################
    # Anet parameters #
    ###################
    lrate = 0.1
    optimizer = 'adagrad'
    haf = 'sigmoid'
    oaf = 'sigmoid'
    loss = 'mean_squared_error'
    hidden_layers = anet_layers[side_length]
    load_existing = False
    anet_name = None


    #########
    # Setup #
    #########

    game = hex.Hex
    game_kwargs = {'side_length': side_length, 'verbose': verbose}

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

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, anet_kwargs=anet_kwargs, train_epochs=train_epochs)

    if play_game:
        list_of_topps = p.play_game(topp=topp_training, topp_k=topp_k)

    game_kwargs = {'side_length': side_length, 'verbose': False}

    if play_versus:
        anet_player = anet.Anet(**anet_kwargs)

        v = versus.Versus(game_kwargs, game, num_versus_matches, player_start, player1=anet_player, player2=None)
        if pre_train:
            anet_player.pre_train(p.get_all_cases())
        v.match()

    if run_topp:
        list_of_topps = ['anet_4x4_topp_0', 'anet_4x4_topp_25', 'anet_4x4_topp_50', 'anet_4x4_topp_75', 'anet_4x4_topp_100']
        topp_ = topp.Topp(list_of_topps, game_kwargs, game, games_per_series)
        topp_.run_topp()
        # topp_.display_scores()

if __name__ == '__main__':
    main()
