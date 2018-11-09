import play
import hex

def main():
    dimensions = 4
    game = hex.Hex
    rollouts = 500
    player_start = 1
    batch_size = 1000
    verbose = False
    nn_model = 'test1'
    pre_train = False

    game_kwargs = {'dimensions': dimensions, 'verbose': verbose}
    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, nn_model=nn_model)
    p.play_game(pre_train=pre_train)

main()

# TODO: Save to file as it trains, play against each other