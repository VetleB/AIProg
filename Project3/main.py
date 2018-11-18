import play
import versus
import hex
import anet

def main():

    anet_layers = {
        3: [120, 64]
        ,4: [120, 64]
        ,5: [200, 128, 64]
    }


    ###################
    # Game parameters #
    ###################
    side_length = 3
    rollouts = (400, 'r')   # r -> amount ; s -> seconds
    player_start = -1       # -1 -> random
    verbose = False

    lrate = 0.01
    optimizer = 'sgd'
    haf = 'sigmoid'
    oaf = 'tanh'
    loss = 'mean_squared_error'
    hidden_layers = anet_layers[side_length]
    load_existing = False
    anet_name = 'test_topp'

    play_game = True
    batch_size = 50
    topp_training = True
    topp_k = 4

    play_versus = False
    num_versus_matches = 1000
    pre_train = True
    pre_train_epochs = 250



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

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, anet_kwargs=anet_kwargs)

    if play_game:
        p.play_game(topp=topp_training, topp_k=topp_k)

    if play_versus:
        anet_player = anet.Anet(**anet_kwargs)
        game_kwargs = {'side_length': side_length, 'verbose': False}

        v = versus.Versus(game_kwargs, game, num_versus_matches, player_start, player1=None, player2=anet_player)
        if pre_train:
            anet_player.pre_train(p.get_all_cases())
        v.match()


if __name__ == '__main__':
    main()
