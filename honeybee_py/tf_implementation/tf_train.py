import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from game.game import tf_create_model

def create_outputs_1other(x1, y1, x2, y2):
    x_dist = x1 - x2
    y_dist = y1 - y2
    if np.abs(x_dist) >= np.abs(y_dist):
        if x_dist > 0:
            move_vector = [0,0,0,1]
        else:
            move_vector = [0,0,1,0]
    else:
        if y_dist > 0:
            move_vector = [1,0,0,0]
        else:
            move_vector = [0,1,0,0]
    return move_vector

def create_outputs_2others(x1, y1, x2, y2, x3, y3, has_food):
    x_dist12 = x1 - x2
    y_dist12 = y1 - y2
    x_dist13 = x1 - x3
    y_dist13 = y1 - y3
    
    if np.abs(x_dist12) >= np.abs(y_dist12):
        if x_dist12 > 0:
            move_vector12 = [0,0,0,1]
        else:
            move_vector12 = [0,0,1,0]
    elif np.abs(x_dist12) < np.abs(y_dist12):
        if y_dist12 > 0:
            move_vector12 = [1,0,0,0]
        else:
            move_vector12 = [0,1,0,0]
            
    if np.abs(x_dist13) >= np.abs(y_dist13):
        if x_dist13 > 0:
            move_vector13 = [0,0,0,1]
        else:
            move_vector13 = [0,0,1,0]
    elif np.abs(x_dist13) < np.abs(y_dist13):
        if y_dist13 > 0:
            move_vector13 = [1,0,0,0]
        else:
            move_vector13 = [0,1,0,0]
    if has_food:
        move_vector = move_vector12
    else:
        move_vector = move_vector13

    return move_vector

def create_positions_1other(board_size=20, size=10000):
    positions = np.random.choice(range(board_size), (size,4))/board_size
    return positions

def create_positions_2others(board_size=20, size=10000):
    positions = np.random.choice(range(board_size), (size,7))/board_size
    positions[:,-1] = np.random.choice([0,1],size)
    return positions

def create_move_set(x, func):
    y = []
    for row in x:
        y.append(func(*row))
    return np.array(y)

if __name__ == "__main__":
    x = create_positions_1other()
    y = create_move_set(x, create_outputs_1other)
    print(x.shape, y.shape)

    model = tf_create_model()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=47)
    
    model.fit(x_train, y_train, batch_size=200, epochs=10, validation_data = (x_val, y_val))
    
    x_test = create_positions_1other(size=2000)
    model_output = model.predict(x_test)
    df = pd.DataFrame(model_output)
    model_output_moves = df.apply(lambda x: np.argmax(x), axis=1)
    correct_moves = []
    for inp in x_test:
        correct_moves.append(np.argmax(create_outputs_1other(*inp)))
    print((model_output_moves == correct_moves).mean())

    model.save('tf_model1.h5')