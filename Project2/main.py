import play
import nim

def main():
    game_kwargs = {'stones': 99, 'move_size': 6, 'verbose': False}
    game = nim.Nim
    rollouts = 400
    player_start = 1
    batch_size = 20
    verbose = False

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size)
    p.play_game()

main()
