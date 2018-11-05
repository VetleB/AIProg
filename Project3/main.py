import play
import hex

def main():
    game_kwargs = {'dimensions': 4, 'verbose': True}
    game = hex.Hex
    rollouts = 400
    player_start = 1
    batch_size = 1
    verbose = False

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size)
    p.play_game()

main()
