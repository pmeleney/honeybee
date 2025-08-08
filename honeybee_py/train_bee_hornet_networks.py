import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import uuid

def move_towards(x1, y1, x2, y2):
    """
    Calculate the direction vector to move from point 1 to point 2.
    
    Args:
        x1 (float): X coordinate of starting point
        y1 (float): Y coordinate of starting point
        x2 (float): X coordinate of target point
        y2 (float): Y coordinate of target point
        
    Returns:
        list: Direction vector [up, down, right, left]
    """
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


def create_outputs_0(x1, y1, x2, y2, x3, y3, has_food):
    """
    Create output vector for regular network (7 inputs).
    
    Args:
        x1, y1 (float): Bee position
        x2, y2 (float): Flower position
        x3, y3 (float): Queen position
        has_food (bool): Whether bee has food
        
    Returns:
        list: Direction vector [up, down, right, left]
    """
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
        move_vector = move_vector13
    else:
        move_vector = move_vector12

    return move_vector

def create_positions_0(board_size = 20, sample_size=10000):
    """
    Create training positions for regular network (7 inputs).
    
    Args:
        board_size (int): Size of the game board
        sample_size (int): Number of training samples to generate
        
    Returns:
        numpy.ndarray: Array of training positions
    """
    positions = np.random.choice(range(board_size), (sample_size,7))/board_size
    positions[:,-1] = np.random.choice([0,1],sample_size)
    
    return positions

def create_outputs_1(x1, y1, x2, y2):
    """
    Create output vector for hornet network (4 inputs).
    
    Args:
        x1, y1 (float): Bee position
        x2, y2 (float): Hornet position
        
    Returns:
        list: Direction vector [up, down, right, left]
    """
    x_dist12 = x1 - x2
    y_dist12 = y1 - y2
    
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

    return move_vector12

def create_positions_1(board_size = 20, sample_size=10000):
    """
    Create training positions for hornet network (4 inputs).
    
    Args:
        board_size (int): Size of the game board
        sample_size (int): Number of training samples to generate
        
    Returns:
        numpy.ndarray: Array of training positions
    """
    positions = np.random.choice(range(board_size), (sample_size,4))/board_size

    
    return positions

def get_wb(num_inputs, epochs, board_size=20, sample_size=100000):
    """
    Train a neural network and save weights and biases.
    
    Args:
        num_inputs (int): Number of inputs (4 for hornet, 7 for regular)
        epochs (int): Number of training epochs
        board_size (int): Size of the game board
        sample_size (int): Number of training samples
        
    Returns:
        keras.Model: Trained model
    """
    if num_inputs == 4:
        x = create_positions_1(board_size=board_size, sample_size=sample_size)
        y = []
        for row in x:
            y.append(create_outputs_1(*row))
        y = np.array(y)
        model_type = 'hornet'
    elif num_inputs == 7:
        x = create_positions_0(board_size=board_size, sample_size=sample_size)
        y = []
        for row in x:
            y.append(create_outputs_0(*row))
        y = np.array(y)
        model_type='regular'
    else:
        raise AttributeError("num_inputs, should be 4 or 7.")
        
    inputs = keras.Input(shape=(num_inputs,))
    l0 = layers.Dense(10, activation="relu")(inputs)
    l1 = layers.Dense(10, activation = "sigmoid")(l0)
    l2 = layers.Dense(10, activation = "sigmoid")(l1)
    outputs = layers.Dense(4, activation = "softmax")(l2)
    model = keras.Model(inputs = inputs, outputs = outputs)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=47)
    model.compile(loss = 'mse', optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.001))
    model.fit(x_train, y_train, batch_size=100, epochs=epochs, validation_data = (x_val, y_val))

    correct_moves = []
    if num_inputs == 4:
        input_array = create_positions_1(board_size,sample_size//5)
        for inp in input_array:
            correct_moves.append(np.argmax(create_outputs_1(*inp)))
    elif num_inputs == 7:
        input_array = create_positions_0(board_size,sample_size//5)
        for inp in input_array:
            correct_moves.append(np.argmax(create_outputs_0(*inp)))
    model_output = model.predict(input_array)
    df = pd.DataFrame(model_output)
    model_output_moves = df.apply(lambda x: np.argmax(x), axis=1)
    print((model_output_moves == correct_moves).mean())

    for layer in model.layers:
        print(len(layer.get_weights()))
        print(layer.get_weights())

    # Create a run-specific subdirectory with timestamp and short id
    os.makedirs('best_weights_and_biases', exist_ok=True)
    run_tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:8]
    run_dir = os.path.join('best_weights_and_biases', f'run_{run_tag}')
    os.makedirs(run_dir, exist_ok=True)

    for i, layer in enumerate(model.layers):
        if len(layer.get_weights())>0:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # Save into run-specific directory
            np.savetxt(os.path.join(run_dir, f'Best_weights_model_{model_type}_layer_{i}.csv'), w, delimiter = ',')
            np.savetxt(os.path.join(run_dir, f'Best_biases_model_{model_type}_layer_{i}.csv'), b, delimiter = ',')
            # And also update root files for backward compatibility
            np.savetxt(os.path.join('best_weights_and_biases', f'Best_weights_model_{model_type}_layer_{i}.csv'), w, delimiter = ',')
            np.savetxt(os.path.join('best_weights_and_biases', f'Best_biases_model_{model_type}_layer_{i}.csv'), b, delimiter = ',')

        #model.save(os.path.join('keras_models', f'{model_type}_model.keras'))

    return model

if __name__ == '__main__':
    model_hornet = get_wb(num_inputs=4, epochs=5)
    model_regular = get_wb(num_inputs=7, epochs=10)