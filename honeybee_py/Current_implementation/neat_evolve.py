import sys
import os
import datetime
import multiprocessing
import pickle
import warnings

import neat
import numpy as np
import matplotlib.pyplot as plt

from game.game import Game, play_game

def eval_genomes(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game()
    genome.fitness = play_game(game, net)
    return genome.fitness

def run(config_file, fdt, restore_checkpoint=None, checkpoint_save_freq=100):
     # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    if restore_checkpoint:
        p = neat.Checkpointer.restore_checkpoint(restore_checkpoint)
    else:
        p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    fdt = format_datetime()
    os.makedirs(os.path.join('checkpoints', fdt))
    p.add_reporter(neat.Checkpointer(checkpoint_save_freq, None, os.path.join('checkpoints', fdt, f'neat_checkpoint_{fdt}_')))
    pe = neat.ParallelEvaluator(2*multiprocessing.cpu_count()-1, eval_genomes)
    winner = p.run(pe.evaluate, None)
    return winner, stats
    
def format_datetime(dt=datetime.datetime.now()):
    formatted_dt = dt.strftime("%Y%m%d%H%M%S")
    return formatted_dt

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

if __name__ == '__main__':
    config_path = sys.argv[1]
    if len(sys.argv) == 3:
        restore_checkpoint = sys.argv[2]
    else:
        restore_checkpoint = None
    fdt = format_datetime()
    winner, stats = run(config_path, fdt, restore_checkpoint)
    plot_stats(stats, filename=os.path.join('outputs','figures',fdt+'_fitness.svg'))
    with open(os.path.join('outputs', 'winners', fdt+'_winner'), 'wb') as f:
        pickle.dump(winner, f)