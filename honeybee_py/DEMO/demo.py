import json
import os

import tensorflow as tf
import numpy as np

from moral_evolution import Network, Config
from game.game import Game, play_game

def demo():
    """
    Run a visualized demonstration of the network as described by the files
    in the demo_files repo.
    """
    # Initialize game
    game = Game()
    # Initialize moral_net
    moral_config_file = os.path.join('net_config', 'morality_layer_config.json')
    moral_config = Config(config_file=moral_config_file)
    moral_net = Network(moral_config)
    # Add previously trained weights and biases to moral_net
    for i, layer in enumerate(moral_net.layers):
            weights = np.loadtxt(os.path.join('demo_files', f"weights_{i}.csv"), dtype=float, delimiter=',')
            moral_net.layers[i].weights = weights
            biases = np.loadtxt(os.path.join('demo_files', f"biases_{i}.csv"), dtype=float, delimiter=',')
            moral_net.layers[i].biases = biases
    # Load TF models
    qf_model = tf.keras.models.load_model(os.path.join('demo_files', 'qf_model.keras'))
    h_model = tf.keras.models.load_model(os.path.join('demo_files', 'h_model.keras'))
    # Make visualization
    play_game(game, moral_net, qf_model, h_model, viz=True, speed=2)
    return None


if __name__ == "__main__":
    demo()