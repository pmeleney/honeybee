# TODO
# - fix slowdown of demonstration
# - add extra bees

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from tensorflow import keras
from scipy.spatial.distance import cosine
from game.gameobjects import Queen, Bee, Flower, Hornet
from game.honeybeeconfig import GameVars, GameState
from game.helpers import _flatten_list, _fill_rect, _distance

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
        self.hornets = self.init_hornets_start(self.game_state.NUM_STARTING_HORNETS)
        self.hornet_exists = False
        self.num_food = 0
        self.game_board = self.update_game()
        self.queen_alive = True
        
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
        positions = _fill_rect(bmxl, bmyu, np.array(self.game_state.QUEEN_SIZE) + 2*np.array(self.game_state.BLANK_MOAT))
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
    
    def create_hornet(self, hornets):
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
        
    def init_hornets_start(self, num_starting_hornets):
        # Initialize starting hornets if exists.
        hornets = []
        if self.game_state.HORNETS_EXIST:
            for _ in range(num_starting_hornets):
                hornets = self.create_hornet(hornets)
        return hornets
    
    def init_hornets_during(self, hornets):
        if ((self.game_vars.turn_num % self.game_state.HORNET_FREQUENCY) == 0) and (self.game_vars.turn_num != 0):
            hornets = self.create_hornet(hornets)
        return hornets

    def place_hornets(self, game_board):
        if self.game_state.HORNETS_EXIST and (len(self.hornets) > 0):
            for hornet in self.hornets:
                game_board[hornet.position[1], hornet.position[0], :] = hornet.color
        return game_board

    def update_game(self):
        """
        Update the gameboard with the new positions of all elements.
        """
        # Static Objects
        game_board = 100*np.ones([self.game_state.NUM_GRID[0], self.game_state.NUM_GRID[1], 3]) #Init to light gray color
        game_board = self.place_queen(game_board)
        game_board = self.place_flowers(game_board)

        # Create hornets list
        if self.game_state.HORNETS_EXIST:
            # Move existing hornets
            if self.hornet_exists:
                if (self.game_vars.turn_num % self.game_state.HORNET_MOVE_SPEED == 0):
                    self.hornets = self.move_hornets()

            if (self.game_vars.turn_num % self.game_state.HORNET_FREQUENCY) == 0:
                self.hornets = self.init_hornets_during(self.hornets)
        # Place Hornets
        game_board = self.place_hornets(game_board)
        
        # Create new bee every time enough food is gathered.
        if (self.num_food % self.game_state.FOOD_PER_BEE == 0) and (self.num_food != 0):
            while len(self.bees) < ((self.num_food/self.game_state.FOOD_PER_BEE)+self.game_state.NUM_STARTING_BEES):
                new_bee = Bee()
                self.bees.append(new_bee)

        #Place Bees
        game_board = self.place_bees(game_board)

        return game_board

    def normalize_outputs(self, net_outputs):
        """
        Normalize outputs to be probabilities with softmax
        """
        a = np.array(net_outputs)
        e_x = np.exp(a - np.max(a))
        return e_x / e_x.sum()

    def find_nearest_bee(self, hornets, bees):
        ## THERES GOT TO BE A BETTER WAY TO DO THIS.
        nearest_bees = {}
        for hornet in hornets:
            bee_distances = {}
            min_distance = np.inf
            for bee in bees:
                distance = _distance(hornet.position, bee.position)
                bee_distances[bee] = distance
            for bee in bee_distances.keys():
                if (bee_distances[bee] < min_distance) and (bee not in nearest_bees.values()):
                    min_distance = bee_distances[bee]
                    nearest_bees[hornet] = bee

        return nearest_bees #dict with hornet as key and bee as value.
            
    def net_move_bees(self, moral_net, qf_model, h_model, nearest_bees):
        """
        Use the evolved network to move each bee.
        """
        nearest_hornets = dict((v,k) for k,v in nearest_bees.items())
        for bee in self.bees:

            # flowers = self.flowers.copy()
            nearest_flower = bee.find_nearest_flower_with_food(self.flowers)
            qf_network_inputs = self.get_qf_inputs(bee, nearest_flower, self.queen)
            qf_outputs = qf_model.predict(qf_network_inputs, verbose=False)

            if bee in nearest_hornets.keys():
                nearest_hornet = nearest_hornets[bee]
                h_network_inputs = self.get_h_inputs(bee, nearest_hornet.position)
                h_outputs = h_model.predict(h_network_inputs, verbose=False)
                moral_output = moral_net.run_once(True)
            else:
                hornet_position = np.array([0,0])
                h_network_inputs = self.get_h_inputs(bee, hornet_position)
                h_outputs = h_model.predict(h_network_inputs, verbose=False)
                moral_output = moral_net.run_once(False)

            if moral_output > 0:
                bee.position = self.net_move(bee, h_outputs)
            else:
                bee.position = self.net_move(bee, qf_outputs)

            overlap = bee.check_overlap(self.queen, self.flowers, self.hornets)
            if overlap == 'Flower' and (np.array(bee.position) == np.array(nearest_flower.position)).all():
                bee.get_food(nearest_flower)
            if overlap == 'Queen':
                bee.drop_food()
                bee.score += 1
                self.num_food += 1
            if overlap == 'Hornet' and (bee.position == nearest_hornet.position).all():
                self.kill_hornet(nearest_hornet) #kill hornet
                bee.score -= 100
        return None
    
    def kill_hornet(self, hornet):
        self.hornets.remove(hornet)
        return None

    def move_hornets(self):
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
    
    def get_h_inputs(self, bee, hornet_position):
        output = np.append(bee.position, hornet_position).T
        return np.array([output])


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
    
    def get_qf_inputs(self, bee, nearest_flower, queen, input_type='positions'):
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


def play_game(game, moral_net, regular_network, hornet_network, viz=False, speed=2):
    images = []
    while game.game_vars.turn_num < game.game_state.MAX_TURNS:
        nearest_bees = game.find_nearest_bee(game.hornets, game.bees)
        game.net_move_bees(moral_net, regular_network, hornet_network, nearest_bees)
        game.game_board = game.update_game()
        images.append(game.game_board/255)
        game.game_vars.turn_num += 1
        if not game.queen_alive:
            break
    images = np.array(images)
    if viz:
        visualize(images, speed)

    return game.queen_alive, game.num_food

def visualize(images, speed):
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')

    app = pg.mkQApp("HoneyBee Demo")

    ## Create window with ImageView widget
    win = QtWidgets.QMainWindow()
    win.resize(800,800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('HoneyBee Game')

    imv.setImage(images)
    imv.play(speed)
    pg.exec()
    return None
    
    

if __name__ == '__main__':
    game = Game()
