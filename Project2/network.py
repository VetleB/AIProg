import node
import random


class Tree:

    def __init__(self, root, game):
        self.root = node.Node(root, None, game)
        self.tree = {root: self.root}
        self.game = game

    # Perform a number of tree searches and rollouts given a root state
    def simulate_game(self, state, rollouts):
        root = self.tree[state]

        for rollout in range(rollouts):
            leaf_node = self.tree_search(root)

            if not leaf_node.win_state():
                # Expansion
                leaf_node.children = []
                child_states = self.game.generate_child_states(leaf_node.state)

                for child_state in child_states:
                    # Add node if it does not already exist
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
        end_value = 1 if end_state[1] == 0 else 0
        update_node.wins += end_value

        # Recursive call
        if update_node.parent is not None:
            self.back_prop(end_state, update_node.parent)

    def rollout(self, leaf_node):
        state = leaf_node.state

        # Choose a random child state
        while not self.game.game_over(state):
            options = self.game.generate_child_states(state)
            choice = random.choice(options)
            state = choice

        return state

    # Search until leaf node or end state
    def tree_search(self, root):
        node = root

        # Manage parents so that backprop works correctly
        while node.children is not None and not node.win_state():
            old_node = node
            node = self.tree_policy(node)
            node.parent = old_node
        return node

    def tree_policy(self, node, expl=True):
        best_node = node.children[0]
        best_val = best_node.value(expl)

        # Find child node with highest value
        for child in node.children:
            val = child.value(expl)
            if val > best_val:
                best_node = child
                best_val = val

        return best_node
