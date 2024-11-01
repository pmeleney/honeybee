import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from game.game import Game, tf_play_game, tf_create_model

if __name__ == '__main__':
    from keras.models import load_model
    model = tf_create_model()

    game = Game()
    model = load_model('tf_model1.h5')
    print(model.summary())
    tf_play_game(game, model, True)