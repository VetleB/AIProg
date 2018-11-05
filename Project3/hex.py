from collections import deque

class Hex:

    def __init__(self, dimensions, player_start, verbose=True):
        if dimensions in range(3, 9):
            self.side_len = dimensions
        else:
            raise ValueError('Size must be between 3x3 and 8x8')

        self.player = player_start

        self.players = {1: 'Red', 0: 'Black'}

        self.state = self.get_initial_state()

        self.verbose = verbose

    def get_initial_state(self):
        empty_board = tuple(['*' for i in range(self.side_len**2)])
        initial_state = (empty_board, self.player)
        return initial_state

    def generate_child_states(self, state):
        child_state_list = []
        state_list = list(state[0])
        player = state[1]

        for i in range(len(state_list)):
            if state_list[i] == '*':
                child_list = state_list[:]
                child_list[i] = str(player)
                child_tuple = tuple(child_list)
                child_state = self.switch_player((child_tuple, player))

                child_state_list.append(child_state)

        return child_state_list

    def make_actual_move(self, state):
        moving_player = self.player_to_string(self.state[1])

        move = None
        board = self.state_deepen(self.state)
        new_board = self.state_deepen(state)
        for i in range(self.side_len):
            for j in range(self.side_len):
                if board[i][j] != new_board[i][j]:
                    move = (i, j)

        if self.verbose:
            print(moving_player, "placed a stone in position", move)
            self.print_hex(state)

        self.state = state


    def player_to_string(self, player):
        return self.players[player]

    def actual_game_over(self):
        end = self.game_over(self.state)

        if end and self.verbose:
            print(self.player_to_string(self.winner(self.state)), "wins")
            print()

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

        if state.count('*') > self.side_len**2 - ((2*self.side_len)-1):
            return False

        board = self.state_deepen(state)

        #self.print_hex(state)

        for i in range(self.side_len):
            # Check for red victory
            if board[0][i] == '1':
                win, searched_board = self.bfs(board, (0, i), 1)
                if win:
                    return True
                board = searched_board

            # Check for black victory
            if board[i][0] == '0':
                win, searched_board = self.bfs(board, (i, 0), 0)
                if win:
                    return True
                board = searched_board

        return False

    def bfs(self, board, root, player):
        dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        row_or_col = 0 if player == 1 else 1

        queue = deque()
        queue.append(root)
        test_board = board[:]
        test_board[root[0]][root[1]] = '*'

        while len(queue):
            root = queue.popleft()

            for direction in dirs:
                invalid = False

                try:
                    x = root[0] + direction[0]
                    y = root[1] + direction[1]

                    if x < 0 or y < 0:
                        invalid = True

                    if test_board[x][y] == str(player) and not invalid:
                        pos = (x, y)
                        if pos[row_or_col] == self.side_len-1:
                            return True, test_board
                        queue.append(pos)
                        test_board[x][y] = '*'
                except IndexError:
                    pass

        return False, test_board

    def state_deepen(self, state):
        board = []

        for i in range(self.side_len):
            row = []
            for j in range(self.side_len):
                element = state[0][self.side_len*i+j]
                row.append(element)
            board.append(row)

        return board

    def player_value(self, state, q, u, expl=True):
        u = u if expl else 0
        if state[1] == 1:
            return -(q-u)
        else:
            return q+u

    def print_header(self):
        self.print_hex(self.state)

    def print_hex(self, state):
        top_diamond = ''
        board = self.state_deepen(state)

        for i in range(1, self.side_len+1):
            top_diamond += ' '*(self.side_len-i)
            x = i
            y = -1
            for j in range(i):
                x -= 1
                y += 1
                top_diamond += str(board[x][y]) + ' '
            top_diamond += '\n'

        board = board[::-1]
        board = [row[::-1] for row in board]

        bottom_diamond = ''

        for i in range(1, self.side_len):
            x = i
            y = -1
            for j in range(i):
                x -= 1
                y += 1
                bottom_diamond += str(board[x][y]) + ' '
            bottom_diamond += ' '*(self.side_len-i-1)
            bottom_diamond += '' if i == self.side_len-1 else '\n'

        bottom_diamond = bottom_diamond[::-1]

        diamond = top_diamond + bottom_diamond

        print(diamond)


