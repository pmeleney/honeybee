import sys
import os
import pickle

import neat

from game.game import Game, play_game

def demo(winner_path, config_path):
    game = Game()
    with open(winner_path, 'rb') as f:
        c = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    net = neat.nn.FeedForwardNetwork.create(c, config)
    play_game(game, net, True)
    return None

if __name__ == '__main__':
      demo(sys.argv[1], sys.argv[2])
        
