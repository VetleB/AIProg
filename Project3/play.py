import network
import random


class Play:

    def __init__(self, game_kwargs, game, num_rollouts, player_start, batch_size=1):
        self.game_kwargs = game_kwargs
        self.game_kwargs['player_start'] = player_start

        self.game_manager = game
        self.game = game(**game_kwargs)

        self.player_start = player_start
        self.rollouts = num_rollouts
        self.batch_size = batch_size

    def play_game(self):
        self.game.print_header()
        print('They will play', self.batch_size, 'games.')
        if self.player_start == -1:
            print('Starting player is random.')
        else:
            print('Player', self.game.player_to_string(self.player_start), 'makes the first move.')

        P1_wins = 0
        for batch in range(self.batch_size):
            self.game_kwargs['player_start'] = self.choose_starting_player()
            self.game = self.game_manager(**self.game_kwargs)
            tree = network.Tree(self.game.get_initial_state(), self.game)

            while not self.game.actual_game_over():
                # Perform tree searches and rollouts
                tree.simulate_game(self.game.state, self.rollouts)
                # Find next actual move
                move_node = tree.tree_policy(tree.tree[self.game.state], expl=False)
                # Make actual move
                self.game.make_actual_move(move_node.state)

            P1_wins += self.game.winner(self.game.state)

        print('P1 wins', P1_wins, 'out of', self.batch_size, 'games (' + str(100*P1_wins/self.batch_size) + ')%')

    def choose_starting_player(self):
        if self.player_start == -1:
            n = random.random()
            return 0 if n < 0.5 else 1
        return self.player_start
