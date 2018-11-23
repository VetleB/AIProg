from collections import deque
import itertools
import random

class Hex:

    def __init__(self, side_length, player_start, verbose=True):
        if side_length in range(3, 9):
            self.side_len = side_length
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

        # Create children by making every legal move on current board with current player
        for i in range(len(state_list)):
            if state_list[i] == '*':
                child_list = state_list[:]
                child_list[i] = str(player)
                child_tuple = tuple(child_list)
                child_state = self.switch_player((child_tuple, player))

                child_state_list.append(child_state)

        return child_state_list

    def make_actual_move(self, state):
        if self.verbose:
            self.print_move(state)

        self.state = state

    def get_move(self, pre_state, post_state):
        move = None
        board = self.state_deepen(pre_state)
        new_board = self.state_deepen(post_state)

        # Figure out move based on difference in states
        for i in range(self.side_len):
            for j in range(self.side_len):
                if board[i][j] != new_board[i][j]:
                    move = (i, j)
        return move

    def print_move(self, state):
        moving_player = self.player_to_string(self.state[1])

        move = self.get_move(self.state, state)

        print(moving_player, "placed a stone in position", move)
        self.print_hex(state)

    # Get move as index usable in lists, etc.
    def get_move_index(self, pre_state, post_state):

        move = self.get_move(pre_state, post_state)

        return move[0]*self.side_len + move[1]

    def get_empty_case(self):
        return [0 for i in range(self.side_len**2)]

    def case_to_nn_feature(self, case):
        nn_board_flat = self.state_to_nn_state(case[0])

        label = case[1]

        return [nn_board_flat, label]

    # Convert game state into feature usable by anet
    def state_to_nn_state(self, state):
        nn_board_positions = {
            '*': [0,0] # Empty
            , '1': [1,0] # Player 1
            , '0': [0,1] # Player 2
        }

        nn_players = {
            1: [1, 0]
            , 0: [0, 1]
        }

        board = list(state[0])
        nn_board = [nn_board_positions[c] for c in board]

        player = state[1]
        nn_player = nn_players[player]

        nn_board.insert(0, nn_player)
        nn_board_flat = list(itertools.chain(*nn_board))

        return nn_board_flat

    # Have an anet choose a child
    def anet_choose_child(self, state, anet):
        position = self.move_index(state, anet)
        return self.make_move(state, position)

    # Turn move index into co-ordinates (to be used for online tournament)
    def anet_choose_move(self, state, anet):
        move_indx = self.move_index(state, anet)

        row = move_indx//self.side_len
        column = move_indx%self.side_len

        move = (row, column)

        return move

    # Based on argument state, return index of move anet wants to make
    def move_index(self, state, anet):
        distribution = anet.distribution([self.state_to_nn_state(state)])
        distribution = list(distribution)[0]
        board = list(state[0])

        # Make illegal moves 0
        for i in range(len(board)):
            if board[i] != '*':
                distribution[i] = 0

        distribution = anet.normalize(distribution)
        position = distribution.index(max(distribution))
        return position

    # Based on a state and a move (as index), return the resulting state after making the move
    def make_move(self, state, move):
        board = list(state[0])

        board[move] = str(state[1])

        new_state = self.switch_player((board, state[1]))

        return new_state

    def request_human_move(self, state):
        move = input("Move: ")

        if move == '':
            return self.request_random_move(state)
        else:
            move = int(move)

        while state[0][move] != '*':
            print('Invalid move')
            move = input("Move: ")
            if move == '':
                return self.request_random_move(state)
            else:
                move = int(move)

        return self.make_move(state, move)

    def request_random_move(self, state):
        moves = []
        for i in range(self.side_len**2):
            if state[0][i] == '*':
                moves.append(i)
        move = random.choice(moves)
        return self.make_move(state, move)

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

    # Check if a state is an end state
    def game_over(self, state):

        if state.count('*') > self.side_len**2 - ((2*self.side_len)-1):
            return False

        board = self.state_deepen(state)

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

    # Perform a breadth-first search to check if a player has connected their sides
    def bfs(self, board, root, player):
        # Possible directions a connection can be made in
        dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        row_or_col = 0 if player == 1 else 1

        queue = deque()
        queue.append(root)
        test_board = board[:]
        # root = (x, y)
        test_board[root[0]][root[1]] = '*'

        while len(queue):
            root = queue.popleft()

            for direction in dirs:
                invalid = False

                try:
                    x = root[0] + direction[0]
                    y = root[1] + direction[1]

                    # To avoid negative indices, as these would start fetching stuff from the end
                    if x < 0 or y < 0:
                        invalid = True

                    if test_board[x][y] == str(player) and not invalid:
                        pos = (x, y)

                        # Check if reached other side
                        if pos[row_or_col] == self.side_len-1:
                            return True, test_board

                        queue.append(pos)
                        test_board[x][y] = '*'
                except IndexError:
                    pass

        return False, test_board

    # Turn 1D list into 2D
    def state_deepen(self, state):
        board = []

        for i in range(self.side_len):
            row = []
            for j in range(self.side_len):
                element = state[0][self.side_len*i+j]
                row.append(element)
            board.append(row)

        return board

    # Combines q and u value depending on which player is being evaluated, in order to do max(value) when comparing moves
    def player_value(self, state, q, u, expl=True):
        u = u if expl else 0
        if state[1] == 1:
            return -(q-u)
        else:
            return q+u

    def print_header(self):
        print("Red vs Black")
        if self.verbose:
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

    def get_file_name(self):
        return 'anet_cases/anet_cases_' + str(self.side_len) + 'x' + str(self.side_len) + '.p'

    def get_player(self, state):
        return state[1]
