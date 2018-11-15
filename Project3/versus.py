import network
import random
import anet
import pickle


class Versus:

    def __init__(self, game_kwargs, game, num_matches, player_start, player1=None, player2=None):
        self.game_kwargs = game_kwargs
        self.game_kwargs['player_start'] = player_start

        self.game_manager = game
        self.game = game(**game_kwargs)

        self.num_matches = num_matches

        self.player_start = player_start

        self.players = {1: player1, 0:player2}

    def match(self):
        self.game.print_header()
        print('They will play', self.num_matches, 'games.')
        if self.player_start == -1:
            print('Starting player is random.')
        else:
            print('Player', self.game.player_to_string(self.player_start), 'makes the first move.')

        P1_wins = 0

        for i in range(self.num_matches):
            self.game_kwargs['player_start'] = self.choose_starting_player()
            self.game = self.game_manager(**self.game_kwargs)

            while not self.game.actual_game_over():
                player = self.players[self.game.get_player(self.game.state)]
                if (isinstance(player, anet.Anet)):
                    move_state = self.game.anet_choose_child(self.game.state, player)
                elif player=='human':
                    move_state = self.game.request_human_move(self.game.state)
                elif player == 'random':
                    move_state = self.game.request_random_move(self.game.state)
                # Make actual move
                self.game.make_actual_move(move_state)

            P1_wins += self.game.winner(self.game.state)

        print('P1 wins', P1_wins, 'out of', self.num_matches, 'games (' + str(100*P1_wins/self.num_matches) + ')%')


    def choose_starting_player(self):
        if self.player_start == -1:
            n = random.random()
            return 0 if n < 0.5 else 1
        return self.player_start
