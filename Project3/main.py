import play
import versus
import hex
import anet

def main():
    side_length = 4
    game = hex.Hex
    rollouts = (500, 'r') # r -> amount ; s -> seconds
    player_start = -1
    batch_size = 1000
    num_matches = 1000
    verbose = False
    game_kwargs = {'side_length': side_length, 'verbose': verbose}

    nn_model = None
    input_layer_size = 2*side_length**2+2
    output_size = side_length**2
    anet_name = 'anet_' + str(side_length) + 'x' + str(side_length)
    anet_kwargs = {'layers': [input_layer_size, 120, 64, output_size]
                , 'haf': 'tanh'
                , 'oaf': 'tanh'
                , 'loss': 'mean_squared_error'
                , 'optimizer': 'SGD'
                , 'model_name': anet_name
                , 'pre_train_epochs': 500}
    pre_train = True
    run_train = False


    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, anet_kwargs=anet_kwargs)
    p.play_game(run_train=run_train, pre_train=pre_train)

    #anet_player = p.anet
    anet_player = anet.Anet(**anet_kwargs)
    game_kwargs = {'side_length': side_length, 'verbose': False}

    v = versus.Versus(game_kwargs, game, num_matches, player_start, player1='random', player2=anet_player)
    v.match()

main()

# TODO: Save to file as it trains, play against each other
