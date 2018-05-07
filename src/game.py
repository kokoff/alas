from random import randint, choice

GREEN = 0  # cooperate with evveryone
RED = 1  # cooperate with eachother
BLUE = 2  # cooperate with everyone else
BROWN = 3  # never cooperate


def getPayoff(coop1, coop2):
    if coop1 and coop2:
        return 3
    elif coop1 and not coop2:
        return 0
    elif not coop1 and coop2:
        return 5
    else:
        return 1


def getCooperation(society, sameSociety):
    return society is GREEN or \
           society is RED and sameSociety or \
           society is BLUE and not sameSociety


def getSocietyPayoff(soc1, soc2):
    sameSociety = soc1 == soc2
    coop1 = getCooperation(soc1, sameSociety)
    coop2 = getCooperation(soc2, sameSociety)

    return getPayoff(coop1, coop2)


class Agent:
    def __init__(self, strategy):
        self.games = 0.0
        self.score = 0
        self.society = randint(0, 3)
        self.strategy = strategy

    def setStrategy(self, strategy):
        self.strategy = strategy

    def __str__(self):
        string = ''
        string += 'fitness=' + str(self.score) + '\n'
        string += 'society=' + str(self.society) + '\n'
        string += 'strategy=' + str(self.strategy) + '\n'
        return string

    def __repr__(self):
        return str(self) + '\n'

    def play(self, other):
        payoff = getSocietyPayoff(self.society, other.society)

        self.games += 1
        self.score += payoff

        return payoff

    def chooseSociety(self, other):
        isScoreBigger = self.score > other.score

        newSoc = self.strategy[16 * isScoreBigger + 4 * self.society + other.society]
        if newSoc != -1:
            self.society = newSoc


class Game:

    def __init__(self, agents, N=1):
        self.agents = agents
        self.iterations = N

    def play(self):
        # reset player scores
        for agent in self.agents:
            agent.score = 0
            agent.games = 0

        for _ in range(self.iterations):
            # choose players randomly
            player1 = choice(self.agents)
            player2 = choice([_ for _ in self.agents if _ is not player1])

            # play game and hand out payoffs
            player1.play(player2)
            player2.play(player1)

            # decide weather to switch societies
            player1.chooseSociety(player2)
            player2.chooseSociety(player1)

        # return score
        return [(p.score,) for p in self.agents]


def main():
    population = [Agent([-1] * 32) for _ in range(2)]
    game = Game(population, 10)
    game.play()
    print population


if __name__ == '__main__':
    main()
