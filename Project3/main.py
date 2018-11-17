import play
import versus
import hex
import anet

def main():
    side_length = 3
    rollouts = (500, 'r')   # r -> amount ; s -> seconds
    player_start = -1       # -1 -> random
    verbose = False

    play_game = False
    batch_size = 100

    play_versus = True
    num_versus_matches = 1000
    pre_train = False
    pre_train_epochs = 250


    game = hex.Hex
    game_kwargs = {'side_length': side_length, 'verbose': verbose}

    input_layer_size = 2*side_length**2+2
    output_size = side_length**2
    anet_name = 'anet_' + str(side_length) + 'x' + str(side_length)
    anet_kwargs = {'layers': [input_layer_size, 120, 64, output_size]
        , 'haf': 'tanh'
        , 'oaf': 'tanh'
        , 'loss': 'mean_squared_error'
        , 'optimizer': 'SGD'
        , 'model_name': anet_name
        , 'pre_train_epochs': pre_train_epochs}


    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, anet_kwargs=anet_kwargs)

    if play_game:
        p.play_game()

    if play_versus:
        anet_player = anet.Anet(**anet_kwargs)
        game_kwargs = {'side_length': side_length, 'verbose': False}

        v = versus.Versus(game_kwargs, game, num_versus_matches, player_start, player1=None, player2=anet_player)
        if pre_train:
            anet_player.pre_train(p.get_all_cases())
        v.match()



main()
