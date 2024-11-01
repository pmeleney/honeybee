from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(47)

class QueenFoodTF:

    def make_inputs(self, board_size, sample_size):
        positions = np.random.choice(range(board_size), (sample_size,7))/board_size
        positions[:,-1] = np.random.choice([0,1],sample_size)
        return positions
    
    def get_correct_outputs(self, x1, y1, x2, y2, x3, y3, has_food):
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
    
    def train_model(self, epochs, input_shape, board_size, sample_size):
        X = self.make_inputs(board_size, sample_size)
        y = np.array([self.get_correct_outputs(*x) for x in X])

        inputs = keras.Input(shape=(input_shape,))
        l0 = keras.layers.Dense(10, activation="relu")(inputs)
        l1 = keras.layers.Dense(10, activation = "sigmoid")(l0)
        l2 = keras.layers.Dense(10, activation = "sigmoid")(l1)
        outputs = keras.layers.Dense(4, activation = "softmax")(l2)
        model = keras.Model(inputs = inputs, outputs = outputs)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=47)
        model.compile(loss = 'mse', optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.001))
        model.fit(X_train, y_train, batch_size=100, epochs=epochs, validation_data = (X_val, y_val))

        return model
    
    def test(self, model, board_size, sample_size):
        oos_sample = self.make_inputs(board_size, sample_size)
        df_oos = pd.DataFrame(oos_sample)

        correct_outputs = []
        for _, row in df_oos.iterrows():
            correct_outputs.append(self.get_correct_outputs(*row))
        arg_max_co = np.array([np.argmax(x) for x in correct_outputs])

        outputs = model.predict(oos_sample)
        arg_max_o = np.array([np.argmax(x) for x in outputs])

        return (arg_max_o == arg_max_co).mean()
    
class HornetTF:

    def make_inputs(self, board_size, sample_size):
        positions = np.random.choice(range(board_size), (sample_size,4))/board_size
        return positions
    
    def get_correct_outputs(self, x1, y1, x2, y2):
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
    
    def train_model(self, epochs, input_shape, board_size, sample_size):
        X = self.make_inputs(board_size, sample_size)
        y = np.array([self.get_correct_outputs(*x) for x in X])

        inputs = keras.Input(shape=(input_shape,))
        l0 = keras.layers.Dense(10, activation="relu")(inputs)
        l1 = keras.layers.Dense(10, activation = "sigmoid")(l0)
        l2 = keras.layers.Dense(10, activation = "sigmoid")(l1)
        outputs = keras.layers.Dense(4, activation = "softmax")(l2)
        model = keras.Model(inputs = inputs, outputs = outputs)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=47)
        model.compile(loss = 'mse', optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.001))
        model.fit(X_train, y_train, batch_size=100, epochs=epochs, validation_data = (X_val, y_val))

        return model
    
    def test(self, model, board_size, sample_size):
        oos_sample = self.make_inputs(board_size, sample_size)
        df_oos = pd.DataFrame(oos_sample)

        correct_outputs = []
        for _, row in df_oos.iterrows():
            correct_outputs.append(self.get_correct_outputs(*row))
        arg_max_co = np.array([np.argmax(x) for x in correct_outputs])

        outputs = model.predict(oos_sample)
        arg_max_o = np.array([np.argmax(x) for x in outputs])

        return (arg_max_o == arg_max_co).mean()

    
if __name__ == "__main__":
    qftf = QueenFoodTF()
    htf = HornetTF()
    
    # print('Training Queen-Food Model: ')
    # qf_model = qftf.train_model(epochs=5, input_shape=7, board_size=20, sample_size=1000000)
    # print('Saving Queen-Food Model: ')
    # qf_model.save('qf_model.keras')
    qf_model = keras.models.load_model('qf_model.keras')
    print('Testing Queen-Food Model: ')
    qf_oos_accuracy = qftf.test(qf_model, board_size=20, sample_size=20000)
    
    print('Training Hornet Model: ')
    h_model = htf.train_model(epochs=3, input_shape=4, board_size=20, sample_size=1000000)
    print('Saving Hornet Model: ')
    h_model.save('h_model.keras')
    print('Testing Hornet Model: ')
    h_oos_accuracy = htf.test(h_model, board_size=20, sample_size=20000)

    print(f'Queen-Food Model OOS accuracy: {qf_oos_accuracy}')
    print(f'Hornet Model OOS accuracy: {h_oos_accuracy}')

