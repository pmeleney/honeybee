# TODO
# - fix slowdown of demonstration
# - add extra bees
# - add incentive to return to queen
# - add hornets

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from scipy.spatial.distance import cosine
from game.gameobjects import Queen, Bee, Flower, Hornet
from game.honeybeeconfig import GameVars, GameState
from game.helpers import _flatten_list, _fill_rect

import itertools

class Game:

    def __init__(self):
        """
        Initialize game with gamestate and gamevars
        """
        self.game_state = GameState()
        self.game_vars = GameVars()
        self.queen_locations = self.get_queen_locations()
        self.queen = self.init_queen(self.game_state.QUEEN_SIZE, self.queen_locations)
        self.bees = self.init_bees(self.game_state.NUM_STARTING_BEES)
        self.flowers = self.init_flowers(self.game_state.NUM_FLOWERS)
        self.hornets = self.init_hornets(self.game_state.NUM_STARTING_HORNETS)
        self.game_board = self.update_game_board()
        self.bee_moves = {}
        self.hornet_moves = {}

    def init_bees(self, num_bees):
        bees = []
        for _ in range(num_bees):
            bee = Bee()
            bees.append(bee)
        return bees

    def place_bees(self, game_board):
        for bee in self.bees:
            game_board[bee.position[1],bee.position[0],:] = bee.color
        return game_board
    
    def get_queen_locations(self):
        if self.game_state.QUEEN_POSITION == 'center':
            xl = self.game_state.NUM_GRID[0]//2 - self.game_state.QUEEN_SIZE[0]//2 + 1
            yu = self.game_state.NUM_GRID[1]//2 - self.game_state.QUEEN_SIZE[1]//2 + 1 
        else:
            raise AttributeError('queen position must be "center".')

        queen_locations = _fill_rect(xl, yu, self.game_state.QUEEN_SIZE)
        return queen_locations
    
    def init_queen(self, size, position):
        queen = Queen(size, position)
        return queen
    
    def place_queen(self, game_board):
        for pos in self.queen.position:
            game_board[pos[1],pos[0],:] = self.queen.color
        return game_board
    
    def get_blank_area_around_queen(self):
        xl = self.game_state.NUM_GRID[0]//2 - self.game_state.QUEEN_SIZE[0]//2 + 1
        yu = self.game_state.NUM_GRID[1]//2 - self.game_state.QUEEN_SIZE[1]//2 + 1 
        bmxl = xl - self.game_state.BLANK_MOAT[0]
        bmyu = yu - self.game_state.BLANK_MOAT[1]
        positions = _fill_rect(bmxl, bmyu, np.array(self.game_state.QUEEN_SIZE) + np.array(self.game_state.BLANK_MOAT))
        return positions

    def init_flowers(self, num_food):
        flowers = []
        poss_fp_x = range(self.game_state.BLANK_RIM, self.game_state.NUM_GRID[0]-self.game_state.BLANK_RIM)
        poss_fp_y = range(self.game_state.BLANK_RIM, self.game_state.NUM_GRID[1]-self.game_state.BLANK_RIM)
        poss_fp = list(itertools.product(poss_fp_x, poss_fp_y))
        blank_area_around_queen = self.get_blank_area_around_queen()
        for blank_location in blank_area_around_queen:
            poss_fp.remove(blank_location)
        inds = range(len(poss_fp))
        flower_inds = np.random.choice(inds, num_food, replace=False)
        flower_positions = []
        for i in flower_inds:
            flower_positions.append(poss_fp[i])
        for pos in flower_positions:
            flowers.append(Flower(position = pos))
        return flowers
    
    def place_flowers(self, game_board):
        for flower in self.flowers:
            game_board[flower.position[1], flower.position[0], :] = flower.color
        return game_board
    
    def init_hornets(self, num_hornets):
        pass

    def place_hornets(self):
        pass

    def update_game_board(self):
        """
        Update the gameboard with the new positions of all elements.
        """
        game_board = 100*np.ones([self.game_state.NUM_GRID[0], self.game_state.NUM_GRID[1], 3]) #Init to light gray color
        game_board = self.place_queen(game_board)
        game_board = self.place_bees(game_board)
        game_board = self.place_flowers(game_board)
        return game_board

    def normalize_outputs(self, net_outputs):
        """
        Normalize outputs to be probabilities with softmax
        """
        a = np.array(net_outputs)
        e_x = np.exp(a - np.max(a))
        return e_x / e_x.sum()

    def net_move_bees(self, net, tf=False):
        """
        Use the evolved network to move each bee.
        """
        for bee in self.bees:
            flowers = self.flowers.copy()
            nearest_flower = bee.find_nearest_flower(flowers)
            net_inputs = self.get_inputs(bee, nearest_flower, self.queen)
            if tf:
                norm_outputs = net.predict(np.expand_dims(np.array(net_inputs),0))
                print(norm_outputs)
            else:
                net_outputs = net.activate(net_inputs)
                norm_outputs = self.normalize_outputs(net_outputs)
            bee.position, move = self.net_move(bee, norm_outputs)

            # self.bee_moves.append(move)

            correct_output = self.get_correct_output(net_inputs)
            self.game_vars.reward += self.get_reward(correct_output, norm_outputs)/self.game_state.MAX_TURNS
            self.game_board = self.update_game_board()

            overlap = bee.check_overlap(self.queen, self.flowers, self.hornets)
            if overlap == 'Flower':
                bee.get_food(nearest_flower)
            if overlap == 'Queen':
                bee.drop_food()
        return move
    
    def get_next_best_move(self, norm_outputs, forbidden_moves):
        if 'up' in forbidden_moves:
            norm_outputs[0] = -1
        if 'dn' in forbidden_moves:
            norm_outputs[1] = -1
        if 'rt' in forbidden_moves:
            norm_outputs[2] = -1
        if 'lt' in forbidden_moves:
            norm_outputs[3] = -1
        net_output_max = np.argmax(norm_outputs)
        d_input_args = {0:'up', 1:'dn', 2:'rt', 3:'lt'}
        move = d_input_args[net_output_max]
        return move
        
    def net_move(self, bee, norm_outputs):
        net_output_max = np.argmax(norm_outputs)
        d_input_args = {0:'up', 1:'dn', 2:'rt', 3:'lt'}
        move = d_input_args[net_output_max]

        #Correct if move puts bee off board
        on_uprt_corner = ((bee.position[0] == (self.game_state.NUM_GRID[0]-1)) and (bee.position[1] == 0))
        on_uplt_corner = ((bee.position[0] == 0) and (bee.position[1] == 0))
        on_dnrt_corner = ((bee.position[0] == (self.game_state.NUM_GRID[0]-1)) and (bee.position[1] == (self.game_state.NUM_GRID[1]-1)))
        on_dnlt_corner = ((bee.position[0] == 0) and (bee.position[1] == (self.game_state.NUM_GRID[1]-1)))
        on_rt_edge = (bee.position[0] == self.game_state.NUM_GRID[0]-1)
        on_lt_edge = (bee.position[0] == 0)
        on_up_edge = (bee.position[1] == 0)
        on_dn_edge = (bee.position[1] == self.game_state.NUM_GRID[0]-1)

        on_corner = (on_uprt_corner or on_uplt_corner or on_dnrt_corner or on_dnlt_corner)
        on_edge = (on_rt_edge or on_lt_edge or on_up_edge or on_dn_edge) and np.logical_not(on_corner)

        forbidden_moves = []
        if on_corner:
            if on_uprt_corner:
                forbidden_moves = ['up', 'rt']
            elif on_uplt_corner:
                forbidden_moves = ['up', 'lt']
            elif on_dnrt_corner:
                forbidden_moves = ['dn', 'rt']
            elif on_dnlt_corner:
                forbidden_moves = ['dn', 'lt']
        if on_edge:
            if on_rt_edge:
                forbidden_moves = ['rt']
            elif on_lt_edge:
                forbidden_moves = ['lt']
            elif on_up_edge:
                forbidden_moves = ['up']
            elif on_dn_edge:
                forbidden_moves = ['dn']

        if move in forbidden_moves:
            move = self.get_next_best_move(norm_outputs, forbidden_moves)

        #move the bee
        if move == 'up':
            bee.position[1] -= 1
        elif move == 'dn':
            bee.position[1] += 1
        elif move == 'rt':
            bee.position[0] += 1
        elif move == 'lt':
            bee.position[0] -= 1
        else:
            raise AttributeError(f'Move {move} not recognized.')
        return bee.position, move
    
    def move_towards(self, bee, other):
        if other.name == 'Queen':
            other.pos = other.position[0]
        else:
            other.pos = other.position
        beex, beey = bee.position
        otherx, othery = other.pos
        x_dist = beex - otherx
        y_dist = beey - othery
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
    
    def get_inputs(self, bee, nearest_flower, queen, input_type='positions'):
        inputs = []
        if input_type == 'vectors':
            move_vector_food = self.move_towards(bee, nearest_flower)
            move_vector_queen = self.move_towards(bee, queen)
            for _ in move_vector_food:
                inputs.append(_)
            inputs.append(int(0))
            for _ in move_vector_queen:
                inputs.append(0)
            inputs = np.array(inputs)

        elif input_type == 'positions':
            inputs.append(bee.position[0])
            inputs.append(bee.position[1])
            inputs.append(nearest_flower.position[0])
            inputs.append(nearest_flower.position[1])
            #inputs.append(0) #.append(hornet.positoin[0])
            #inputs.append(0) #.append(hornet.positoin[1])
            #inputs.append(float(bee.has_food)*self.game_board.shape[0])
            #inputs.append(0) #float(self.hornet_exists)*self.game_board.shape[0]            
            inputs = ((np.array(inputs)/self.game_board.shape[0]))
        return inputs
    
    def get_correct_output(self, input, input_type='rel_positions'):
        if input_type == 'rel_positions':
            x_dist = input[0]
            y_dist = input[1]
        elif input_type == 'positions':
            beex = input[0]
            beey = input[1]
            #if bool(input[7]): #bee.has_food()
            otherx = input[2]
            othery = input[3]
            #else:
            #    otherx = input[4]
            #    othery = input[5]
            x_dist = beex - otherx
            y_dist = beey - othery
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
        return np.array(move_vector)

    
    def get_reward(self, correct_outputs, outputs, tf=True):
        if tf:
            return 0.0
        reward = 1.0-cosine(outputs, correct_outputs)
        return reward

    def move_bee(self, bee, output):
        d_input_args = {0:'up', 1:'dn', 2:'rt', 3:'lt'}
        move = d_input_args[output]
        if move == 'up':
            bee.position[1] -= 1
        elif move == 'dn':
            bee.position[1] += 1
        elif move == 'rt':
            bee.position[0] += 1
        elif move == 'lt':
            bee.position[0] -= 1
        return bee.position, move



def play_game(game, net, viz=False):
    if viz:
        plt.ion()
        figure, ax = plt.subplots(figsize=(10,10))

    while game.game_vars.turn_num < game.game_state.MAX_TURNS:
        game.net_move_bees(net)
        game.game_vars.turn_num += 1

        if viz:
            visualize(game, figure, ax)
    return game.game_vars.reward

def tf_play_game(game, net, viz=False):
    if viz:
        plt.ion()
        figure, ax = plt.subplots(figsize=(10,10))
    while game.game_vars.turn_num < game.game_state.MAX_TURNS:
        game.net_move_bees(net, True)
        game.game_vars.turn_num += 1

        if viz:
            visualize(game, figure, ax)

def tf_create_model():
    inputs = keras.Input(shape=(4,))
    l0 = layers.Dense(10, activation="relu")(inputs)
    l1 = layers.Dense(10, activation = "relu")(l0)
    l2 = layers.Dense(10, activation = "sigmoid")(l1)
    outputs = layers.Dense(4, activation = "sigmoid")(l2)
    model = keras.Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate = 0.0025))
    return model

def visualize(game, figure, ax):
    img = game.game_board/255
    ax.imshow(img, interpolation='nearest')
    figure.canvas.draw()
    figure.canvas.flush_events()

if __name__ == '__main__':
    game = Game()
