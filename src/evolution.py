from deap import base, creator
from game import Game, Agent
import random
from deap import tools
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt
import csv
import pandas as pd
import numpy as np

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 32
NUM_IND = 100

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, a=0, b=3)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=-1, up=3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def multiAgentEvolution(CXPB, MUTPB, NGEN, N):
    pop = toolbox.population(n=NUM_IND)
    agents = [Agent(ind) for ind in pop]
    game = Game(agents, N)

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

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)

        # Set new (evolved) genotypes
        for agent, ind in zip(agents, offspring):
            agent.setStrategy(ind)

        # Evaluate the individuals
        fitnesses = game.play()
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    # get best agent
    best = agents[0]
    for agent in agents:
        if best.score < agent.score:
            best = agent

    return best


def singleAgentEvolution(CXPB, MUTPB, NGEN, N):
    pop = toolbox.population(n=NUM_IND)
    agents = [Agent(ind) for ind in pop]
    game = Game(agents, N)
    avg_fitness = []

    # Evaluate the entire population
    fitnesses = game.play()
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select individual to be evolved
        index = random.randint(0, len(pop) - 1)
        # Select the new generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        child1 = offspring[index]
        for child2 in [element for idx, element in enumerate(pop) if idx != index]:
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                continue

        mutant = offspring[index]
        if random.random() < MUTPB:
            toolbox.mutate(mutant)

        # Set new (evolved) genotype
        agents[index].setStrategy(pop[index])

        # The chosen individual is replaced by the offspring
        pop[index] = offspring[index]

        # Evaluate the individuals
        fitnesses = game.play()
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # save average fitness
        avg_fitness.append(np.max(fitnesses))

    plt.plot(avg_fitness)
    plt.show()

    # get best agent
    best = agents[0]
    for agent in agents:
        if best.score < agent.score:
            best = agent

    return best


def test_run(best, N, test_runs):
    print 'TEST RUN'
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


if __name__ == '__main__':
    CXPB, MUTPB, NGEN, N = 0.5, 0.2, 10000, 1000
    best = multiAgentEvolution(CXPB, MUTPB, NGEN, N)
    test_run(best, N, 1000)
