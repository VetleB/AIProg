import play
import nim

def main():
    game_kwargs = {'stones': 39, 'move_size': 3, 'verbose': True}
    game = nim.Nim
    rollouts = 400
    player_start = 1
    batch_size = 1
    verbose = False

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size)
    p.play_game()

main()
