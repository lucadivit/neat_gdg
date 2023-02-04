"""
An experiment using NEAT to perform the simple XOR task.
Fitness threshold set in config
- by default very high to show the high possible accuracy of the NEAT library.
"""

import pickle
import neat
import neat.nn
from pureples.shared.visualize import draw_net
import random, sys

# seed = random.randrange(sys.maxsize)
seed = 2072571730719029873
random.seed(seed)

path = "data/neat/"


# Network inputs and expected outputs.
XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]


def heaviside(z):
    if z >= 0:
        return 1
    else:
        return 0


# Config for FeedForwardNetwork.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            f'{path}config_neat_xor_gdg')
CONFIG.genome_config.add_activation("heaviside", heaviside)


def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness, in this case, as the accuracy
    """
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_case = len(XOR_OUTPUTS)
        counter_true = 0
        for xor_inputs, xor_expected in zip(XOR_INPUTS, XOR_OUTPUTS):
            xor_output = net.activate(xor_inputs)
            if xor_output[0] == xor_expected[0]:
                counter_true += 1
        genome.fitness = counter_true / total_case


def run(gens):
    """
    Create the population and run the XOR task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(eval_fitness, gens)
    print("neat_xor done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    WINNER = run(300)[0]  # Only relevant to look at the winner.
    print('\nBest genome:\n{!s}'.format(WINNER))

    # Verify network output against training data.
    print('\nOutput:')
    WINNER_NET = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)

    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = WINNER_NET.activate(inputs)
        print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # Save net if wished reused and draw it to a file.
    with open(f'{path}winner_neat_xor.pkl', 'wb') as output:
        pickle.dump(WINNER_NET, output, pickle.HIGHEST_PROTOCOL)
    draw_net(WINNER_NET, filename=f"{path}neat_xor_winner")

# print("Seed is", seed)