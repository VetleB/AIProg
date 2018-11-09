import play
import hex

def main():
    dimensions = 4
    game = hex.Hex
    rollouts = 500
    player_start = -1
    batch_size = 10
    verbose = False

    game_kwargs = {'dimensions': dimensions, 'verbose': verbose}
    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size)
    p.play_game()

main()
