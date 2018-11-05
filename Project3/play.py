import hex
import network
import random


class Play:

    def __init__(self, dimensions, num_rollouts, player_start, batch_size=1, verbose=True):
        self.dimensions = dimensions
        self.player_start = player_start

        self.rollouts = num_rollouts
        self.batch_size = batch_size
        self.verbose = verbose

    def play_game(self):
        white_wins = 0
        for batch in range(self.batch_size):
            self.game = hex.Hex(self.dimensions, self.choose_starting_player(), self.verbose)
            tree = network.Tree(self.game.get_initial_state(), self.game)

            while not self.game.actual_game_over():
                tree.simulate_game(self.game.state, self.rollouts)
                move_node = tree.tree_policy(tree.tree[self.game.state], expl=False)
                # options = [(n.state, n.ratio()) for n in move_node.parent.children]
                # print(options)
                self.game.make_actual_move(move_node.state)
                # print()
            white_wins += self.game.winner(self.game.state)
            #print('Game', batch+1, 'winner:', self.game.winner(self.game.state))
        print('Red wins', white_wins, 'out of', self.batch_size, 'games (' + str(100*white_wins/self.batch_size) + ')%')

    def choose_starting_player(self):
        if self.player_start == -1:
            n = random.random()
            return 0 if n < 0.5 else 1
        return self.player_start