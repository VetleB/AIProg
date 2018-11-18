import anet

class Topp:

    def __init__(self, list_of_topps):
        self.agents = []

        for name in list_of_topps:
            agent = anet.Anet(layers=[], model_name=name, load_existing=True)
            self.agents.append(agent)

    def run_topp(self):
        pass
