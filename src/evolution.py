from deap import base, creator
from game import Game, Agent
import random
from deap import tools
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import csv
import pandas as pd

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 32
NUM_IND = 20

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, a=0, b=3)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=-1, up=3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def evolution(CXPB, MUTPB, NGEN, N):
    pop = toolbox.population(n=NUM_IND)
    agents = [Agent(ind) for ind in pop]
    game = Game(agents, N)
    logger = Logger()

    # Evaluate the entire population
    fitnesses = game.play()
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Set new (evolved) genotype
        for agent, ind in zip(agents, offspring):
            agent.setStrategy(ind)

        # Evaluate the individuals with an invalid fitness
        fitnesses = game.play()
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    #     logger.log_societies(agents)
    #
    # logger.write_societies()
    # logger.plot_societies()
    #
    # get best agent
    best = agents[0]
    for agent in agents:
        if best.score < agent.score:
            best = agent

    return best


def singleAgentEvolution(CXPB, MUTPB, NGEN, N):
    pop = toolbox.population(n=10)
    agents = [Agent(ind) for ind in pop]
    game = Game(agents, N)
    logger = Logger()

    # Evaluate the entire population
    fitnesses = game.play()
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = pop
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        mutant = offspring[0]
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Set new (evolved) genotype
        for agent, ind in zip(agents, offspring):
            agent.setStrategy(ind)

        # Evaluate the individuals with an invalid fitness
        fitnesses = game.play()
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # log results
    #     logger.log_societies(agents)
    #
    # logger.write_societies()
    # logger.plot_societies()

    return agents[0]


def test_run(best, N, test_runs):
    results = []
    position = []

    for _ in range(test_runs):
        agents = [Agent([random.randint(-1, 3) for _ in range(IND_SIZE)])
                  for _ in range(NUM_IND)]

        agents[0] = best
        game = Game(agents, N)

        game.play()
        results.append(best.score)
        position.append(sorted([a.score for a in agents], reverse=True).index(best.score))

    plt.hist(results)
    plt.show()

    plt.hist(position)
    plt.show()


class Logger:
    def __init__(self):
        self.societies = {'Green': [], 'Red': [], 'Blue': [], 'Brown': []}
        self.best_fitness = []

    def log_societies(self, agents):
        societies = [agent.society for agent in agents]
        societyCounts = Counter(societies)

        self.societies['Green'].append(societyCounts.pop(0, 0))
        self.societies['Red'].append(societyCounts.pop(1, 0))
        self.societies['Blue'].append(societyCounts.pop(2, 0))
        self.societies['Brown'].append(societyCounts.pop(3, 0))

    def write_societies(self):
        keys = ['Green', 'Red', 'Blue', 'Brown']
        with open('societies.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(zip(*[self.societies[key] for key in keys]))

    def plot_societies(self):
        df = pd.DataFrame(self.societies)
        df.iloc[::1000, :].plot(kind='bar')
        plt.show()


if __name__ == '__main__':
    CXPB, MUTPB, NGEN, N = 0.5, 0.2, 5000, 1000
    best = evolution(CXPB, MUTPB, NGEN, N)
    test_run(best, N, 1000)
