import node
import random
import time


class Tree:

    def __init__(self, root, game, anet, epsilon=0.1):
        self.root = node.Node(root, None, game)
        self.tree = {root: self.root}
        self.game = game
        self.anet = anet
        self.epsilon = epsilon

    def simulate_game(self, state, rollouts):
        root = self.tree[state]

        # Decide if rollouts are time-based or num-based
        if rollouts[1] == 'r':
            self.rollouts_amount(root, rollouts[0])
        elif rollouts[1] == 's':
            self.rollouts_time(root, rollouts[0])

        # Initialize empty distribution
        distribution = self.game.get_empty_case()

        # Put the visit counts in the right place in the distribution
        for child in root.children:
            index = self.game.get_move_index(root.state, child.state)
            distribution[index] = child.games

        # Normalize distribution
        distribution = self.anet.normalize(distribution)

        case = (root.state, distribution)
        return case

    # Perform rollouts based on a given amount
    def rollouts_amount(self, root, rollouts):
        for rollout in range(rollouts):
            # Find a leaf node without children (or end-state)
            leaf_node = self.tree_search(root)

            if not leaf_node.win_state():
                leaf_node.children = []
                child_states = self.game.generate_child_states(leaf_node.state)
                for child_state in child_states:
                    # Assume child state already exists, if not create it
                    try:
                        child_node = self.tree[child_state]
                    except KeyError:
                        child_node = node.Node(child_state, leaf_node, self.game)
                        self.tree[child_state] = child_node

                    leaf_node.children.append(child_node)

                # Perform rollout on one of the newly created child nodes
                child_rollout_node = self.tree_search(leaf_node)
                end_state = self.rollout(child_rollout_node)

                # Propagate information back up from the child node
                self.back_prop(end_state, child_rollout_node)
            else:
                self.back_prop(leaf_node.state, leaf_node)

    # Perform rollouts based on a given time
    def rollouts_time(self, root, time_per_move):
        start_time = time.time()
        while time.time()-start_time < time_per_move:
            # Find a leaf node without children (or end-state)
            leaf_node = self.tree_search(root)

            if not leaf_node.win_state():
                leaf_node.children = []
                child_states = self.game.generate_child_states(leaf_node.state)
                for child_state in child_states:
                    # Assume child state already exists, if not create it

                    try:
                        child_node = self.tree[child_state]
                    except KeyError:
                        child_node = node.Node(child_state, leaf_node, self.game)
                        self.tree[child_state] = child_node

                    leaf_node.children.append(child_node)

                # Perform rollout on one of the newly created child nodes
                child_rollout_node = self.tree_search(leaf_node)
                end_state = self.rollout(child_rollout_node)

                # Propagate information back up from the child node
                self.back_prop(end_state, child_rollout_node)
            else:
                self.back_prop(leaf_node.state, leaf_node)

    def back_prop(self, end_state, update_node):
        update_node.games += 1
        end_value = 1 if self.game.winner(end_state) == 1 else 0
        update_node.wins += end_value

        if update_node.parent is not None:
            self.back_prop(end_state, update_node.parent)

    # Given a state, have anet choose new state, used for the 'random' part of rollouts
    def rollout(self, leaf_node):
        state = leaf_node.state

        while not self.game.game_over(state):
            if random.random() > self.epsilon:
                state = self.game.anet_choose_child(state, self.anet)
            else:
                options = self.game.generate_child_states(state)
                choice = random.choice(options)
                state = choice

        return state

    # Perform a tree search from root to leaf node
    def tree_search(self, root):
        node = root
        while node.children is not None and not node.win_state():
            old_node = node
            node = self.tree_policy(node)
            node.parent = old_node
        return node

    # Choose a node's highest value child
    def tree_policy(self, node, expl=True):
        best_node = node.children[0]
        best_val = best_node.value(expl)

        for child in node.children:
            val = child.value(expl)
            if val > best_val:
                best_node = child
                best_val = val

        return best_node
