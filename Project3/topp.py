import anet
import versus


class Topp:

    def __init__(self, list_of_topps, game_kwargs, game, games_per_series=10):
        self.game_kwargs = game_kwargs
        self.game_kwargs['player_start'] = -1

        self.game = game(**game_kwargs)

        self.games = games_per_series

        self.versus_match = versus.Versus(game_kwargs, game, games_per_series, player_start=-1)

        self.agents = []

        # Load all topp competitors
        for name in list_of_topps:
            agent = anet.Anet(layers=[], model_name=name, load_existing=True)
            self.agents.append(agent)

        self.agent_scores = {}

        for agent in self.agents:
            self.agent_scores[agent] = 0

    def run_topp(self):
        for agent1 in range(len(self.agents)-1):
            for agent2 in range(agent1+1, len(self.agents)):
                player1 = self.agents[agent1]
                player2 = self.agents[agent2]

                self.versus_match.players[1] = player1
                self.versus_match.players[0] = player2

                scores = self.versus_match.match(verbose=False)

                self.agent_scores[player1] += scores[0]
                self.agent_scores[player2] += scores[1]

                print(player1.file_name, 'vs', player2.file_name)
                print(scores[0], '-', scores[1])

    def display_scores(self):
        for agent in self.agents:
            agent_name = agent.file_name
            agent_score = self.agent_scores[agent]

            print(agent_name, agent_score)
