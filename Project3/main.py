import play
import versus
import hex

def main():
    dimensions = 4
    game = hex.Hex
    rollouts = 500
    player_start = 0
    batch_size = 1
    num_matches = 1
    verbose = True
    nn_model = 'test1'
    pre_train = False
    run_train = False

    game_kwargs = {'dimensions': dimensions, 'verbose': verbose}
    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size, nn_model=nn_model)
    p.play_game(run_train=run_train, pre_train=pre_train)

    anet_player = p.anet

    v = versus.Versus(game_kwargs, game, num_matches, player_start, player2=anet_player)
    v.match()

main()

# TODO: Save to file as it trains, play against each other
