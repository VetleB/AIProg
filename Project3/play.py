import network
import random
import anet
import pickle


class Play:

    def __init__(self, game_kwargs, game, rollouts, player_start, batch_size, anet_kwargs, pre_train_epochs=250):
        self.game_kwargs = game_kwargs
        self.game_kwargs['player_start'] = player_start

        self.game_manager = game
        self.game = game(**game_kwargs)

        self.player_start = player_start
        self.rollouts = rollouts
        self.batch_size = batch_size

        self.anet = anet.Anet(**anet_kwargs)
        self.pre_train_epochs = pre_train_epochs

    def play_game(self, run_train=True, pre_train=False):
        self.game.print_header()
        print('They will play', self.batch_size, 'games.')
        if self.player_start == -1:
            print('Starting player is random.')
        else:
            print('Player', self.game.player_to_string(self.player_start), 'makes the first move.')

        P1_wins = 0
        try:
            with open(self.game.get_file_name(), 'rb') as f:
                try:
                    all_cases = pickle.load(f)
                except:
                    all_cases = []
        except (FileNotFoundError, EOFError):
            f = open(self.game.get_file_name(), 'w')
            all_cases = []

        if pre_train:
            self.pre_train(epochs=self.pre_train_epochs)
            self.anet.accuracy(all_cases)

        for batch in range(self.batch_size):
            self.game_kwargs['player_start'] = self.choose_starting_player()
            self.game = self.game_manager(**self.game_kwargs)

            if run_train:
                tree = network.Tree(self.game.get_initial_state(), self.game, self.anet)
                rbuf = []

            while not self.game.actual_game_over():
                if run_train:
                    # Perform tree searches and rollouts
                    case = tree.simulate_game(self.game.state, self.rollouts)
                    nn_case = self.game.case_to_nn_feature(case)
                    rbuf.append(nn_case)

                    # Find next actual move
                    move_node = tree.tree_policy(tree.tree[self.game.state], expl=False)
                    move_state = move_node.state
                else:
                    move_state = self.game.anet_choose_child(self.game.state, self.anet)

                # Make actual move
                self.game.make_actual_move(move_state)

            if run_train:
                random.shuffle(rbuf)
                self.anet.train_on_cases(rbuf)
                all_cases.extend(rbuf)

            P1_wins += self.game.winner(self.game.state)

        if run_train:
            self.anet.accuracy(all_cases)

            self.anet.save_model()

        with open(self.game.get_file_name(), 'wb') as f:
            pickle.dump(all_cases, f)

        print('P1 wins', P1_wins, 'out of', self.batch_size, 'games (' + str(100*P1_wins/self.batch_size) + ')%')

    def choose_starting_player(self):
        if self.player_start == -1:
            n = random.random()
            return 0 if n < 0.5 else 1
        return self.player_start

    def pre_train(self, epochs):
        with open(self.game.get_file_name(), 'rb') as f:
            cases = pickle.load(f)
            self.anet.pre_train(cases)
