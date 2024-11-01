import neat
import os

def eval_genomes(genomes, config):
    


def run_neat(config):
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50) #max generations = 50



if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    config.path = os.path.join(dirpath + config.txt)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                         config_path)
    
    run_neat(config)