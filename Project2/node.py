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
        q = self.wins/self.games if self.games > 0 else 0.5
        return q

    def u(self):
        N = self.parent.games
        num = math.log(N) if N > 1 else 1
        u = math.sqrt(num/(1+self.games))
        return u

    def win_state(self):
        return self.game.game_over(self.state)

    # Pass Q and u value to the game and let it figure out how to combine them.
    def value(self, expl=True):
        return self.game.player_value(self.state, self.Q(), self.u(), expl)

    def ratio(self):
        numerator = str(self.wins)
        denominator = str(self.games)
        return numerator+'/'+denominator
