from collections import deque

class Hex:

    def __init__(self, dim, player_start, verbose=True):
        if dim in range(3, 9):
            self.side_len = dim
        else:
            raise ValueError('Size must be between 3x3 and 8x8')

        self.player = player_start

        self.players = {1: 'Red', 0: 'Black'}

        self.state = self.get_initial_state()

        self.verbose = verbose

    def get_initial_state(self):
        empty_board = tuple([-1 for i in range(self.side_len**2)])
        initial_state = (empty_board, self.player)
        return initial_state

    def generate_child_states(self, state):
        child_state_list = []
        state_list = list(state[0])
        player = state[1]

        for i in range(len(state_list)):
            if state_list[i] == -1:
                child_state = state_list[:]
                child_state[i] = player
                child_state_list.append((tuple(child_state), player))

        return child_state_list

    def make_actual_move(self, state):
        moving_player = self.player_to_string(self.state[1])

        move = None
        for i in range(self.side_len):
            for j in range(self.side_len):
                if self.state[i][j] != state[i][j]:
                    move = (i, j)

        if self.verbose:
            print(moving_player, "placed a stone in position", move)

        self.state = state

    def player_to_string(self, player):
        return self.players[player]

    def actual_game_over(self):
        end = self.game_over(self.state)

        if end and self.verbose:
            print(self.player_to_string(self.winner(self.state)), "wins")

        return end

    def winner(self, state):
        if self.game_over(state):
            return int(not state[1])
        return None

    def switch_player(self, state):
        new_player = int(not state[1])
        new_state = (state[0], new_player)
        return new_state

    def game_over(self, state):

        if state.count(-1) < (2*self.side_len)-1:
            return False

        board = self.state_deepen(state)

        for i in range(self.side_len):
            # Check for red victory
            if board[0][i] == 0:
                win, searched_board = self.bfs(board, (0, i), 0)
                if win:
                    return True
                board = searched_board

            # Check for black victory
            if board[i][0] == 1:
                win, searched_board = self.bfs(board, (i, 0), 1)
                if win:
                    return True
                board = searched_board

        return False

    def bfs(self, board, root, player):
        dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        row_or_col = player

        queue = deque()
        queue.append(root)
        board[root[0]][root[1]] = -1

        while len(queue):
            root = queue.popleft()

            for direction in dirs:
                try:
                    x = root[0] + direction[0]
                    y = root[1] + direction[1]

                    if board[x][y] == player:
                        pos = (x, y)
                        if pos[row_or_col] == self.side_len-1:
                            return True, board
                        queue.append(pos)
                        board[x][y] = -1
                except IndexError:
                    pass

        return False, board

    def state_deepen(self, state):
        board = []

        for i in range(self.side_len):
            row = []
            for j in range(self.side_len):
                element = state[self.side_len*i+j]
                row.append(element)
            board.append(row)

        return board

    def player_value(self, state, q, u, expl=True):
        u = u if expl else 0
        if state[1] == 1:
            return -(q-u)
        else:
            return q+u
