import play
import nim

def main():
    game_kwargs = {'stones': 100, 'move_size': 23, 'verbose': False}
    game = nim.Nim
    rollouts = 100
    player_start = -1
    batch_size = 10
    verbose = False

    p = play.Play(game_kwargs, game, rollouts, player_start, batch_size)
    p.play_game()
