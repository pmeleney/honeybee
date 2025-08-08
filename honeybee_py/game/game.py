# TODO
# - fix slowdown of demonstration
# - add extra bees

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from scipy.spatial.distance import cosine
from .gameobjects import Queen, Bee, Flower, Hornet
from .honeybeeconfig import GameVars, GameState
from .helpers import _flatten_list, _fill_rect

import itertools
class Game:
    """
    Main game class that manages the honeybee simulation.
    
    This class handles the game board, all game objects (bees, flowers, hornets, queen),
    neural network interactions, and game state management.
    """

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
        self.hornet_exists = False
        self.game_board = self.update_game_board()
        self.bee_moves = {}
        self.hornet_moves = {}
        self.queen_alive = True
        

    def init_bees(self, num_bees):
        """
        Initialize a list of bees.
        
        Args:
            num_bees (int): Number of bees to create
            
        Returns:
            list: List of Bee objects
        """
        bees = []
        for _ in range(num_bees):
            bee = Bee()
            bees.append(bee)
        return bees

    def place_bees(self, game_board):
        """
        Place bees on the game board.
        
        Args:
            game_board (numpy.ndarray): The game board array
            
        Returns:
            numpy.ndarray: Updated game board with bees placed
        """
        for bee in self.bees:
            game_board[bee.position[1],bee.position[0],:] = bee.color
        return game_board
    
    def get_queen_locations(self):
        """
        Calculate the queen's position on the board.
        
        Returns:
            list: List of coordinate tuples for the queen's position
            
        Raises:
            AttributeError: If queen position is not 'center'
        """
        if self.game_state.QUEEN_POSITION == 'center':
            xl = self.game_state.NUM_GRID[0]//2 - self.game_state.QUEEN_SIZE[0]//2 + 1
            yu = self.game_state.NUM_GRID[1]//2 - self.game_state.QUEEN_SIZE[1]//2 + 1 
        else:
            raise AttributeError('queen position must be "center".')

        queen_locations = _fill_rect(xl, yu, self.game_state.QUEEN_SIZE)
        return queen_locations
    
    def init_queen(self, size, position):
        """
        Initialize the queen bee.
        
        Args:
            size (tuple): Size of the queen as (width, height)
            position (list): List of coordinate tuples for queen's position
            
        Returns:
            Queen: Queen object
        """
        queen = Queen(size, position)
        return queen
    
    def place_queen(self, game_board):
        """
        Place the queen on the game board.
        
        Args:
            game_board (numpy.ndarray): The game board array
            
        Returns:
            numpy.ndarray: Updated game board with queen placed
        """
        for pos in self.queen.position:
            game_board[pos[1],pos[0],:] = self.queen.color
        return game_board
    
    def get_blank_area_around_queen(self):
        """
        Get the blank area around the queen where flowers cannot be placed.
        
        Returns:
            list: List of coordinate tuples for the blank area
        """
        xl = self.game_state.NUM_GRID[0]//2 - self.game_state.QUEEN_SIZE[0]//2 + 1
        yu = self.game_state.NUM_GRID[1]//2 - self.game_state.QUEEN_SIZE[1]//2 + 1 
        bmxl = xl - self.game_state.BLANK_MOAT[0]
        bmyu = yu - self.game_state.BLANK_MOAT[1]
        positions = _fill_rect(bmxl, bmyu, np.array(self.game_state.QUEEN_SIZE) + np.array(self.game_state.BLANK_MOAT))
        return positions

    def init_flowers(self, num_food):
        """
        Initialize flowers with random positions, avoiding the queen's area.
        
        Args:
            num_food (int): Number of flowers to create
            
        Returns:
            list: List of Flower objects
        """
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
        """
        Place flowers on the game board.
        
        Args:
            game_board (numpy.ndarray): The game board array
            
        Returns:
            numpy.ndarray: Updated game board with flowers placed
        """
        for flower in self.flowers:
            game_board[flower.position[1], flower.position[0], :] = flower.color
        return game_board
    
    def init_hornets(self, num_starting_hornets):
        """
        Initialize hornets with random positions on the board edges.
        
        Args:
            num_starting_hornets (int): Number of hornets to create
            
        Returns:
            list: List of Hornet objects
        """
        if self.game_state.HORNETS_EXIST:
            hornets = []
            for _ in range(num_starting_hornets):
                horiz_boundary = np.random.choice([0,1])
                if horiz_boundary:
                    x_coord = int(np.floor(np.random.choice(range(self.game_state.NUM_GRID[0]))))
                    y_coord = int(np.random.choice([0,1]) * (self.game_state.NUM_GRID[1]-1))
                else:
                    y_coord = int(np.floor(np.random.choice(range(self.game_state.NUM_GRID[1]))))
                    x_coord = int(np.random.choice([0,1]) * (self.game_state.NUM_GRID[0]-1))
                hornet = Hornet(position=(x_coord, y_coord))
                hornets.append(hornet)
                self.hornet_exists = True
            
            if ((self.game_vars.turn_num % self.game_state.HORNET_FREQUENCY) == 0) and (self.game_vars.turn_num != 0):
                horiz_boundary = np.random.choice([0,1])
                if horiz_boundary:
                    x_coord = int(np.floor(np.random.choice(range(self.game_state.NUM_GRID[0]))))
                    y_coord = int(np.random.choice([0,1]) * (self.game_state.NUM_GRID[1]-1))
                else:
                    y_coord = int(np.floor(np.random.choice(range(self.game_state.NUM_GRID[1]))))
                    x_coord = int(np.random.choice([0,1]) * (self.game_state.NUM_GRID[0]-1))
                hornet = Hornet(position=(x_coord, y_coord))
                hornets.append(hornet)
                self.hornet_exists = True
        return hornets

    def place_hornets(self, game_board):
        """
        Place hornets on the game board.
        
        Args:
            game_board (numpy.ndarray): The game board array
            
        Returns:
            numpy.ndarray: Updated game board with hornets placed
        """
        if self.game_state.HORNETS_EXIST:
            if len(self.hornets) > 0:
                for hornet in self.hornets:
                    game_board[hornet.position[1], hornet.position[0], :] = hornet.color
        return game_board


    def update_game_board(self):
        """
        Update the gameboard with the new positions of all elements.
        """
        game_board = 100*np.ones([self.game_state.NUM_GRID[0], self.game_state.NUM_GRID[1], 3]) #Init to light gray color
        game_board = self.place_queen(game_board)
        game_board = self.place_bees(game_board)
        game_board = self.place_flowers(game_board)
        if self.game_state.HORNETS_EXIST:
            if np.logical_not(self.hornet_exists):
                self.hornets = self.init_hornets(self.game_state.NUM_STARTING_HORNETS)
            if (self.game_vars.turn_num % 2 == 0) and (self.game_vars.turn_num != 0):
                self.hornets = self.move_hornet()
            game_board = self.place_hornets(game_board)
        return game_board

    def normalize_outputs(self, net_outputs):
        """
        Normalize outputs to be probabilities with softmax
        """
        a = np.array(net_outputs)
        e_x = np.exp(a - np.max(a))
        return e_x / e_x.sum()

    def net_move_bees(self, moral_net, regular_net, hornet_net, movement_type='regular'):
        """
        Use the evolved network to move each bee.
        """
        for bee in self.bees:
            flowers = self.flowers.copy()
            nearest_flower = bee.find_nearest_flower_with_food(flowers)
            regular_network_inputs = self.get_regular_inputs(bee, nearest_flower, self.queen)
            if len(self.hornets) > 0:
                hornet_position = np.array(self.hornets[0].position)
            else:
                hornet_position = np.array([self.queen.position[0][0],self.queen.position[0][1]])
            hornet_network_inputs = self.get_hornet_inputs(bee, hornet_position)
            regular_outputs = regular_net.predict(regular_network_inputs, verbose=False)
            hornet_outputs = hornet_net.predict(hornet_network_inputs, verbose=False)
            moral_output = moral_net.run_once(self.hornet_exists)
            moral_decision = float(np.squeeze(moral_output))
            if moral_decision > 0.5:
                bee.position = self.net_move(bee, hornet_outputs)
            else:
                bee.position = self.net_move(bee, regular_outputs)

            overlap = bee.check_overlap(self.queen, self.flowers, self.hornets)
            if overlap == 'Flower' and (np.array(bee.position) == np.array(nearest_flower.position)).all():
                bee.get_food(nearest_flower)
            if overlap == 'Queen':
                bee.drop_food()
                bee.score += 1
            if overlap == 'Hornet':
                self.hornets = [] #kill hornet
                self.hornet_exists = False
                bee.score -= 100
        return None
    
    def move_hornet(self):
        """
        Move hornets towards the queen.
        
        Returns:
            list: Updated list of hornets
        """
        for hornet in self.hornets:
            vector = self.move_towards(hornet, self.queen)
            d_input_args = {0:'up', 1:'dn', 2:'rt', 3:'lt'}
            v_arg_max = np.argmax(np.array(vector))
            move = d_input_args[v_arg_max]
            if move == 'up':
                hornet.position[1] -= 1
            elif move == 'dn':
                hornet.position[1] += 1
            elif move == 'rt':
                hornet.position[0] += 1
            elif move == 'lt':
                hornet.position[0] -= 1

            overlap = hornet.check_overlap(self.queen)
            if overlap:
                self.queen_alive = False

        return self.hornets
    
    def get_next_best_move(self, norm_outputs, forbidden_moves):
        """
        Get the next best move when the preferred move is forbidden.
        
        Args:
            norm_outputs (numpy.ndarray): Normalized network outputs
            forbidden_moves (list): List of forbidden move directions
            
        Returns:
            str: The next best move direction ('up', 'dn', 'rt', 'lt')
        """
        if (norm_outputs.shape[1]) > 1:
            norm_outputs = norm_outputs[0]
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
        """
        Move a bee based on neural network outputs.
        
        Args:
            bee (Bee): The bee to move
            norm_outputs (numpy.ndarray): Normalized network outputs
            
        Returns:
            list: New position of the bee
        """
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
        return bee.position
    
    def get_hornet_inputs(self, bee, hornet_position):
        """
        Get inputs for the hornet network.
        
        Args:
            bee (Bee): The bee object
            hornet_position (numpy.ndarray): Position of the hornet
            
        Returns:
            numpy.ndarray: Input array for the hornet network
        """
        output = np.append(bee.position, hornet_position).T
        return np.array([output])


    def move_towards(self, bee, other):
        """
        Calculate the direction vector to move towards another object.
        
        Args:
            bee (Bee): The bee object
            other (GameObject): The target object to move towards
            
        Returns:
            list: Direction vector [up, down, right, left]
        """
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
    
    def get_regular_inputs(self, bee, nearest_flower, queen, input_type='positions'):
        """
        Get inputs for the regular network.
        
        Args:
            bee (Bee): The bee object
            nearest_flower (Flower): The nearest flower with food
            queen (Queen): The queen object
            input_type (str): Type of input encoding ('positions' or 'vectors')
            
        Returns:
            numpy.ndarray: Input array for the regular network
        """
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
            inputs.append(queen.position[0][0])
            inputs.append(queen.position[0][1])
            
            #inputs.append(0) #float(self.hornet_exists)*self.game_board.shape[0]         
            inputs = (np.array(inputs)/self.game_board.shape[0])
            inputs = np.concatenate([inputs, np.array([float(bee.has_food)])])
            
        return np.array([inputs])
    
    def get_correct_output(self, input, input_type='rel_positions'):
        """
        Calculate the correct output for given inputs.
        
        Args:
            input (numpy.ndarray): Input array
            input_type (str): Type of input encoding
            
        Returns:
            numpy.ndarray: Correct output vector
        """
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
        """
        Calculate reward based on output correctness.
        
        Args:
            correct_outputs (numpy.ndarray): Correct outputs
            outputs (numpy.ndarray): Actual outputs
            tf (bool): Whether using TensorFlow
            
        Returns:
            float: Reward value
        """
        if tf:
            return 0.0
        reward = 1.0-cosine(outputs, correct_outputs)
        return reward

    def move_bee(self, bee, output):
        """
        Move a bee based on output direction.
        
        Args:
            bee (Bee): The bee to move
            output (int): Output direction index
            
        Returns:
            tuple: (new_position, move_direction)
        """
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


def play_game(game, moral_net, regular_network, hornet_network, viz=False):
    """
    Play a complete game with the given networks.
    
    Args:
        game (Game): Game object
        moral_net: Moral network
        regular_network: Regular network
        hornet_network: Hornet network
        viz (bool): Whether to visualize the game
        
    Returns:
        tuple: (queen_alive, bee_score)
    """
    if viz:
        plt.ion()
        figure, ax = plt.subplots(figsize=(10,10))

    while game.game_vars.turn_num < game.game_state.MAX_TURNS:
        game.net_move_bees(moral_net, regular_network, hornet_network)
        game.game_board = game.update_game_board()
        game.game_vars.turn_num += 1

        if viz:
            visualize(game, figure, ax)
    return game.queen_alive, game.bees[0].score

def tf_create_model():
    """
    Create a TensorFlow model for the game.
    
    Returns:
        keras.Model: Compiled TensorFlow model
    """
    inputs = keras.Input(shape=(4,))
    l0 = layers.Dense(10, activation="relu")(inputs)
    l1 = layers.Dense(10, activation = "relu")(l0)
    l2 = layers.Dense(10, activation = "sigmoid")(l1)
    outputs = layers.Dense(4, activation = "sigmoid")(l2)
    model = keras.Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate = 0.0025))
    return model

def visualize(game, figure, ax):
    """
    Visualize the current game state.
    
    Args:
        game (Game): Game object
        figure (matplotlib.figure.Figure): Figure object
        ax (matplotlib.axes.Axes): Axes object
    """
    img = game.game_board/255
    ax.imshow(img, interpolation='nearest')
    figure.canvas.draw()
    figure.canvas.flush_events()

if __name__ == '__main__':
    game = Game()
