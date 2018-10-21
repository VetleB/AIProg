import nim
import network


class Play:

    def __init__(self, stones, move_size, player_start, num_rollouts):

        self.game = nim.Nim(stones, move_size, player_start)
        self.rollouts = num_rollouts

    def play_game(self, episodes):
        for episode in range(episodes):
            tree = network.Tree(self.game.get_initial_state(), self.game)

            while not self.game.actual_game_over():
                tree.simulate_game(self.game.state, self.rollouts)
                move_node = tree.tree_policy(tree.tree[self.game.state], expl=False)
                options = [(n.state, n.ratio()) for n in move_node.parent.children]
                print(options)
                self.game.make_actual_move(move_node.state)
                #print()
