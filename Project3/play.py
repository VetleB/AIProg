import network
import random
import anet
import pickle


class Play:

    def __init__(self, game_kwargs, game, rollouts, player_start, batch_size, rbuf_mbs, anet_kwargs, train_epochs=250):
        self.game_kwargs = game_kwargs
        self.game_kwargs['player_start'] = player_start

        self.game_manager = game
        self.game = game(**game_kwargs)

        self.player_start = player_start
        self.rollouts = rollouts
        self.batch_size = batch_size
        self.rbuf_mbs = rbuf_mbs

        self.anet = anet.Anet(**anet_kwargs)
        self.anet.save_model()
        self.train_epochs = train_epochs

    def play_game(self, topp=False, topp_k=4):
        self.game.print_header()
        print('They will play', self.batch_size, 'games.')
        if self.player_start == -1:
            print('Starting player is random.')
        else:
            print('Player', self.game.player_to_string(self.player_start), 'makes the first move.')

        P1_wins = 0

        # Fetch cases if there are any
        all_cases = self.get_all_cases()

        # Set up saving of anets throughout the run
        if topp:
            topp_list = []

            topp_save_batches = []
            topp_interval = self.batch_size // (topp_k-1)

            for i in range(0, self.batch_size, topp_interval):
                topp_save_batches.append(i)
            topp_save_batches.append(self.batch_size)

        # Clear replay buffer
        rbuf = []

        for batch in range(self.batch_size):

            if topp and (batch in topp_save_batches):
                topp_name = self.anet.topp_save(batch)
                topp_list.append(topp_name)

            self.game_kwargs['player_start'] = self.choose_starting_player()

            # Initialize game and MC tree
            self.game = self.game_manager(**self.game_kwargs)
            tree = network.Tree(self.game.get_initial_state(), self.game, self.anet)

            while not self.game.actual_game_over():
                # Perform tree searches, rollouts and backprops
                case = tree.simulate_game(self.game.state, self.rollouts)
                nn_case = self.game.case_to_nn_feature(case)

                # Add case to rbuf
                rbuf.append(nn_case)

                # Find next actual move
                move_node = tree.tree_policy(tree.tree[self.game.state], expl=False)
                move_state = move_node.state

                # Make actual move
                self.game.make_actual_move(move_state)

            # Randomize rbuf
            random.shuffle(rbuf)

            # Train on minibatch of rbuf cases
            self.anet.train_on_cases(rbuf[0:self.rbuf_mbs], epochs=self.train_epochs)

            # print(self.game_kwargs['player_start'] == self.game.winner(self.game.state))

            P1_wins += self.game.winner(self.game.state)

            print(str(batch+1) + '/' + str(self.batch_size))

        # Add rbuf to all cases
        all_cases.extend(rbuf)

        # Write all cases to file
        with open(self.game.get_file_name(), 'wb') as f:
            pickle.dump(all_cases, f)

        #self.anet.train_on_cases(rbuf, epochs=self.train_epochs)
        #print(len(rbuf))
        # self.anet.accuracy(all_cases)

        print('P1 wins', P1_wins, 'out of', self.batch_size, 'games (' + str(100*P1_wins/self.batch_size) + ')%')

        if topp:
            topp_name = self.anet.topp_save(self.batch_size)
            topp_list.append(topp_name)
            return topp_list
        else:
            self.anet.save_model()
            return []

    def choose_starting_player(self):
        if self.player_start == -1:
            n = random.random()
            return 0 if n < 0.5 else 1
        return self.player_start

    def get_all_cases(self):
        try:
            with open(self.game.get_file_name(), 'rb') as f:
                try:
                    all_cases = pickle.load(f)
                except:
                    all_cases = []
        except (FileNotFoundError, EOFError):
            all_cases = []

        return all_cases
