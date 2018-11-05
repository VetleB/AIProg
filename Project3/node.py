import math

class Node:

    def __init__(self, state, parent, game):
        self.games = 0
        self.wins = 0
        self.children = None
        self.parent = parent
        self.state = state
        self.game = game

    def Q(self):
        return self.wins/self.games

    def u(self):
        N = self.parent.games
        num = math.log(N) if N > 1 else 1
        u = math.sqrt(num/(self.games))
        return u if u != 0 else 0.5

    def win_state(self):
        return self.game.game_over(self.state)

    def value(self, expl=True):
        return self.game.player_value(self.state, self.Q(), self.u(), expl)

    def ratio(self):
        numerator = str(self.wins)
        denominator = str(self.games)
        return numerator+'/'+denominator
