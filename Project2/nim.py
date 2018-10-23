class Nim:

    def __init__(self, stones, move_size, player_start):
        if stones > 2:
            self.stones = stones
        else:
            raise ValueError('Number of stones must be at least 3')

        if move_size in range(2, stones):
            self.move_size = move_size
        else:
            raise ValueError('Maximum size of move must be greater than 1 but less than # of stones')

        self.player = player_start

        self.players = {1: 'White', 0: 'Black'}

        self.state = self.get_initial_state()

    def get_initial_state(self):
        initial_state = (self.stones, self.player)
        return initial_state

    def generate_child_states(self, state):
        u_lim = self.move_size if state[0] >= self.move_size else state[0]
        child_state_list = []

        for i in range(u_lim):
            child_state_list.append(self.make_simulated_move(state, i+1))

        return child_state_list

    def make_actual_move(self, state):
        moving_player = self.player_to_string(self.state[1])
        num_rocks_taken = self.state[0] - state[0]
        remaining = state[0]

        print(moving_player, "picks", num_rocks_taken, "stones: Remaining stones =", remaining)

        self.state = state

    def make_simulated_move(self, state, num_stones_taken):
        stones = state[0]

        if not self.game_over(state):
            if num_stones_taken in range(1, self.move_size+1) and num_stones_taken <= stones:
                state = self.take_stones(state, num_stones_taken)
                state = self.switch_player(state)
            else:
                return None
        else:
            return None
        return state

    def player_to_string(self, player):
        return self.players[player]

    def actual_game_over(self):
        end = self.game_over(self.state)

        if end:
            print(self.player_to_string(self.winner(self.state)), "wins")

        return end

    def winner(self, state):
        if self.game_over(state):
            return int(not state[1])
        return None

    def take_stones(self, state, stones_taken):
        new_stones = state[0]-stones_taken
        new_state = (new_stones, state[1])
        return new_state

    def switch_player(self, state):
        new_player = int(not state[1])
        new_state = (state[0], new_player)
        return new_state

    def game_over(self, state):
        if state[0] == 0:
            return True
        return False

    def player_value(self, state, q, u, expl=True):
        u = u if expl else 0
        if state[1] == 1:
            return -(q-u)
        else:
            return q+u
