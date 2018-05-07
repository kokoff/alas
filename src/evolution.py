from deap import base, creator
from game import Game, Agent
from matplotlib import pyplot as plt


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

import random
from deap import tools

IND_SIZE = 32

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, a=0, b=3)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    return sum(individual),


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=-1, up=3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def main():
    CXPB, MUTPB, NGEN, N = 0.5, 0.2, 40, 100

    pop = toolbox.population(n=10)
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

        # # plot histogram
        # fig = plt.gcf()
        # fig.clear()
        # soc = [agent.society for agent in agents]
        # plt.hist(soc)
        # plt.show()

    print pop
    print fitnesses
    return pop

if __name__ == '__main__':
    main()
