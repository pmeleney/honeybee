import numpy as np
rng = np.random.default_rng()
from .game.game import Game, play_game
import json
import os
from scipy.spatial import distance
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_INPUTS = 100
NUM_NETS = 10

def write_se_config():
    """
    Write configuration files for the moral evolution experiment.
    
    Creates three JSON configuration files:
    - morality_layer_config.json: Configuration for the moral decision network
    - regular_network_config.json: Configuration for the regular behavior network
    - hornet_network_config.json: Configuration for the hornet confrontation network
    """
    morality_layer_config = {
        # Run Attributes
        'reward_percentile' : 0,
        'reward_function' : 'percentage_correct_reward',
        'stopping_criteria' : 0.95,
        'init_new' : True,
        'board_size' : 20,
        'num_inputs' : NUM_INPUTS,
        'max_num_iter_without_improvement' : 8,

        # Net Attributes
        'net_shape' : [[1,1]],
        'layer_activations' : ['softmax'],
        'num_nets' : NUM_NETS,

        # Exponent Capping
        'exp_cap' : 100,

        # Weights Attributes
            # Init        
        'init_weights_dist' : 'normal',
        'init_weights_uniform_range' : 2.0,
        'init_weights_normal_loc' : 0.0,
        'init_weights_normal_scale' : 1.0,

            # Update        
        'update_weights_dist' : 'normal',
        'update_weights_uniform_range' : 2.0,
        'update_weights_normal_loc' : 0.0,
        'update_weights_normal_scale' : 1.0,

        # Biases Attributes
            # Init
        'init_biases_dist' : 'normal',
        'init_biases_uniform_range' : 2.0,
        'init_biases_normal_loc' : 0.0,
        'init_biases_normal_scale' : 1.0,
        
            # Update
        'update_biases_dist' : 'normal',
        'update_biases_uniform_range' : 2.0,
        'update_biases_normal_loc' : 0.0,
        'update_biases_normal_scale' : 1.0,
    }

    regular_network_config = {
        'name' : 'regular',
        'reward_function' : None,
        'board_size' : 20,
        'num_inputs' : NUM_INPUTS,
        'exp_cap' : 100,

        'net_shape' : [[7,10], [10,10], [10,10], [10,4]],
        'layer_activations' : ['relu', 'sigmoid', 'sigmoid', 'softmax'],
        'weights_files' : [os.path.join('best_weights_and_biases', 'Best_weights_model_regular_layer_1.csv'),
                           os.path.join('best_weights_and_biases', 'Best_weights_model_regular_layer_2.csv'),
                           os.path.join('best_weights_and_biases', 'Best_weights_model_regular_layer_3.csv'),
                           os.path.join('best_weights_and_biases', 'Best_weights_model_regular_layer_4.csv')],
        'biases_files' : [os.path.join('best_weights_and_biases', 'Best_biases_model_regular_layer_1.csv'),
                          os.path.join('best_weights_and_biases', 'Best_biases_model_regular_layer_2.csv'),
                          os.path.join('best_weights_and_biases', 'Best_biases_model_regular_layer_3.csv'),
                          os.path.join('best_weights_and_biases', 'Best_biases_model_regular_layer_4.csv')],

    }

    hornet_network_config = {
        'name' : 'hornet',
        'reward_function': None,
        'board_size' : 20,
        'num_inputs' : NUM_INPUTS,
        'exp_cap' : 100,

        'net_shape' : [[4,10], [10,10], [10,10], [10,4]],
        'layer_activations' : ['relu', 'sigmoid', 'sigmoid', 'softmax'],
        'weights_files' : [os.path.join('best_weights_and_biases', 'Best_weights_model_hornet_layer_1.csv'),
                           os.path.join('best_weights_and_biases', 'Best_weights_model_hornet_layer_2.csv'),
                           os.path.join('best_weights_and_biases', 'Best_weights_model_hornet_layer_3.csv'),
                           os.path.join('best_weights_and_biases', 'Best_weights_model_hornet_layer_4.csv')],
        'biases_files' : [os.path.join('best_weights_and_biases', 'Best_biases_model_hornet_layer_1.csv'),
                          os.path.join('best_weights_and_biases', 'Best_biases_model_hornet_layer_2.csv'),
                          os.path.join('best_weights_and_biases', 'Best_biases_model_hornet_layer_3.csv'),
                          os.path.join('best_weights_and_biases', 'Best_biases_model_hornet_layer_4.csv')],
        
    }

    with open(os.path.join('config_files', 'morality_layer_config.json'), 'w') as f:
        json.dump(morality_layer_config, f)

    with open(os.path.join('config_files', 'regular_network_config.json'), 'w') as f:
        json.dump(regular_network_config, f)

    with open(os.path.join('config_files', 'hornet_network_config.json'), 'w') as f:
        json.dump(hornet_network_config, f)

    return None

class Config():
    """
    Configuration class for loading and managing network parameters.
    
    This class loads configuration from JSON files and provides access to
    network parameters, training settings, and reward functions.
    """

    def __init__(self, config_file):
        """
        Initialize configuration from a JSON file.
        
        Args:
            config_file (str): Path to the configuration JSON file
        """
        with open(config_file, 'r') as f:
            config = json.load(f) # To Do

            if config_file == os.path.join('config_files', 'morality_layer_config.json'):
                self.name = 'morality_layer'
                self.reward_function = config['reward_function']
                self.reward_percentile = config['reward_percentile']
                self.stopping_criteria = config['stopping_criteria']
                self.init_new = config['init_new']
                self.board_size = config['board_size']
                self.num_inputs = config['num_inputs']
                self.max_num_iter_without_improvement = config['max_num_iter_without_improvement']
                
                # Net Attributes
                self.net_shape = config['net_shape']
                self.layer_activations = config['layer_activations']
                self.num_nets = config['num_nets']

                self.exp_cap = config['exp_cap']
                
                # Weights Attributes
                    # Init
                self.init_weights_dist = config['init_weights_dist']
                self.init_weights_uniform_range = config['init_weights_uniform_range']
                self.init_weights_normal_loc = config['init_weights_normal_loc']
                self.init_weights_normal_scale = config['init_weights_normal_scale']
                    # Update
                self.update_weights_dist = config['update_weights_dist']
                self.update_weights_uniform_range = config['update_weights_uniform_range']
                self.update_weights_normal_loc = config['update_weights_normal_loc']
                self.update_weights_normal_scale = config['update_weights_normal_scale']
                
                # Biases Attributes
                    # Init
                self.init_biases_dist = config['init_biases_dist']
                self.init_biases_uniform_range = config['init_biases_uniform_range']
                self.init_biases_normal_loc = config['init_weights_normal_loc']
                self.init_biases_normal_scale = config['init_weights_normal_scale']
                    # Update
                self.update_biases_dist = config['update_biases_dist']
                self.update_biases_uniform_range = config['update_biases_uniform_range']
                self.update_biases_normal_loc = config['update_biases_normal_loc']
                self.update_biases_normal_scale = config['update_biases_normal_scale']

            else:
                self.name = config['name']
                self.num_inputs = config['num_inputs']
                self.board_size = config['board_size']
                self.net_shape = config['net_shape']
                self.layer_activations = config['layer_activations']
                self.weights_files = config['weights_files']
                self.biases_files = config['biases_files']
                self.reward_function = config['reward_function']
                self.exp_cap = config['exp_cap']

            return None

class Layer:
    """
    Represents a single layer in a neural network.
    
    This class handles the weights, biases, and activation functions for
    a neural network layer, including initialization, updates, and loading
    from pre-trained models.
    """

    def __init__(self, config, layer_num):
        """
        Initialize a layer with configuration and layer number.
        
        Args:
            config (Config): Configuration object containing layer parameters
            layer_num (int): Index of this layer in the network
        """
        afs = ActivationFunctions(config)

        self.layer_shape = config.net_shape[layer_num]
        if config.name in ['hornet', 'regular']:
            self.weights = self.load_weights(layer_num, config)
            self.biases = self.load_biases(layer_num, config)
        else:
            self.weights = self.init_weights(config)
            self.biases = self.init_biases(config)
        self.activation_function_name = config.layer_activations[layer_num]
        self.activation_function = afs.activation_functions[config.layer_activations[layer_num]]

        return None
    
    def init_weights(self, config):
        """
        Initialize weights of Layer using parameters described in config.
        
        Args:
            config (Config): Configuration object with weight initialization parameters
            
        Returns:
            numpy.ndarray: Initialized weight matrix
            
        Raises:
            AttributeError: If init_weights_dist is not 'uniform' or 'normal'
        """
        if config.init_weights_dist == 'uniform':
            layer_weights = config.init_weights_uniform_range * ((2*np.random.random([self.layer_shape[0], self.layer_shape[1]])) - 1)
        elif config.init_weights_dist == 'normal':
            layer_weights = ((rng.normal(loc=config.init_weights_normal_loc, scale=config.init_weights_normal_scale, size=[self.layer_shape[0], self.layer_shape[1]])))
        else:
            raise AttributeError(f'init_weights_dist not understood: {config.init_weights_dist}.')
        return layer_weights
    
    def init_biases(self, config):
        """
        Initialize biases of Layer using parameters described in config.
        
        Args:
            config (Config): Configuration object with bias initialization parameters
            
        Returns:
            numpy.ndarray: Initialized bias vector
            
        Raises:
            AttributeError: If init_biases_dist is not 'uniform' or 'normal'
        """
        if config.init_biases_dist == 'uniform':
            layer_biases = config.init_biases_uniform_range * ((2*np.random.random(self.layer_shape[1])) - 1)
        elif config.init_biases_dist == 'normal':
            layer_biases = ((rng.normal(loc=config.init_biases_normal_loc, scale=config.init_biases_normal_scale, size=[self.layer_shape[1]])))
        else:
            raise AttributeError(f'init_biases_dist not understood: {config.init_biases_dist}.')
        return layer_biases
    
    def update_weights(self, config, weights_normal_scale):
        """
        Update weights according to configuration parameters.
        
        Args:
            config (Config): Configuration object with weight update parameters
            weights_normal_scale (float): Scale factor for normal distribution updates
            
        Returns:
            numpy.ndarray: Updated weight matrix
            
        Raises:
            AttributeError: If update_weights_dist is not 'uniform' or 'normal'
        """
        if config.update_weights_dist == 'uniform':
            update_to_weights = config.update_weights_uniform_range * ((2*np.random.random([self.layer_shape[0], self.layer_shape[1]])) - 1)
        elif config.update_weights_dist == 'normal':
            update_to_weights = rng.normal(loc=config.update_weights_normal_loc, scale=weights_normal_scale, size=[self.layer_shape[0], self.layer_shape[1]])
        else:
            raise AttributeError(f'config.update_weights_dist not understood: {config.update_weights_dist}.')
        return self.weights + update_to_weights
    
    def update_biases(self, config, biases_normal_scale):
        """
        Update biases according to configuration parameters.
        
        Args:
            config (Config): Configuration object with bias update parameters
            biases_normal_scale (float): Scale factor for normal distribution updates
            
        Returns:
            numpy.ndarray: Updated bias vector
        """
        if config.update_biases_dist == 'uniform':
            update_to_biases = config.update_biases_uniform_range * ((2*np.random.random(self.layer_shape[1])) - 1)
        elif config.update_biases_dist == 'normal':
            update_to_biases = rng.normal(loc=config.update_biases_normal_loc, scale=biases_normal_scale, size=[self.layer_shape[1]])
        return self.biases + update_to_biases
    
    def load_weights(self, layer_num, config):
        """
        Load weights from a CSV file.
        
        Args:
            layer_num (int): Layer number for file naming
            config (Config): Configuration object with weights file path
            
        Returns:
            numpy.ndarray: Loaded weight matrix
        """
        layer_weights = np.loadtxt(config.weights_files[layer_num], delimiter=',', dtype=float) 
        return layer_weights

    def load_biases(self, layer_num, config):
        """
        Load biases from a CSV file.
        
        Args:
            layer_num (int): Layer number for file naming
            config (Config): Configuration object with biases file path
            
        Returns:
            numpy.ndarray: Loaded bias vector
        """
        layer_biases = np.loadtxt(config.biases_files[layer_num], delimiter=',', dtype=float) 
        return layer_biases
    
    def copy(self, config, layer_num):
        """
        Create a copy of this layer.
        
        Args:
            config (Config): Configuration object
            layer_num (int): Layer number
            
        Returns:
            Layer: A new layer with copied weights and biases
        """
        new_layer = Layer(config, layer_num)
        new_layer.weights = self.weights.copy()
        new_layer.biases = self.biases.copy()
        return new_layer

class Network:
    """
    Represents a complete neural network with multiple layers.
    
    This class manages a collection of layers and provides methods for
    forward propagation, training, and network updates.
    """

    def __init__(self, config):
        """
        Initialize a network with configuration.
        
        Args:
            config (Config): Configuration object containing network parameters
        """
        self.layers = []
        for layer_num in range(len(config.net_shape)):
            layer = Layer(config, layer_num)
            self.layers.append(layer)
        self.reward_function = self.select_reward_function(config)
        return None
    
    def copy(self, config):
        """
        Create a copy of this network.
        
        Args:
            config (Config): Configuration object
            
        Returns:
            Network: A new network with copied layers
        """
        new_net = Network(config)
        for layer_num, layer in enumerate(self.layers):
            new_net.layer = layer.copy(config, layer_num)
        return new_net
    
    def select_reward_function(self, config):
        """
        Select the appropriate reward function based on configuration.
        
        Args:
            config (Config): Configuration object
            
        Returns:
            function or None: Selected reward function or None if not specified
        """
        if config.reward_function is not None:
            rfs = RewardFunctions(config)
            return rfs.reward_function
        else:
            return None
    
    def run_once(self, input):
        """
        Run a single forward pass through the network.
        
        Args:
            input: Input to the network (can be bool or numpy array)
            
        Returns:
            numpy.ndarray: Network output
        """
        if type(input) == bool:
            prev_output = np.array([[np.float64(input)]])
            output = (prev_output * self.layers[0].weights) + self.layers[0].biases
        else:
            prev_output = input.copy()
            for layer in self.layers:
                prev_output = (np.matmul(prev_output, layer.weights) + layer.biases)
                if layer.activation_function_name != 'softmax':
                    prev_output = np.array([layer.activation_function(x) for x in prev_output])
                else:
                    prev_output = layer.activation_function(prev_output)
            output = prev_output.copy()
        return output

    def run(self, inputs, display_results=False):
        """
        Run the network on multiple inputs and calculate average reward.
        
        Args:
            inputs (numpy.ndarray): Array of inputs
            display_results (bool): Whether to print results
            
        Returns:
            float: Average reward across all inputs
        """
        reward = 0
        for _, input in enumerate(inputs):
            output = self.run_once(input)
            correct_output = self.get_correct_output(input)
            reward += self.reward_function(output, correct_output)
            if display_results:
                print(output, correct_output, reward)
        return reward/(inputs.shape[0])
    
    def get_outputs(self, inputs, display_results=False):
        """
        Get outputs for multiple inputs without calculating rewards.
        
        Args:
            inputs (numpy.ndarray): Array of inputs
            display_results (bool): Whether to print results
            
        Returns:
            numpy.ndarray: Array of network outputs
        """
        outputs = []
        for _, input in enumerate(inputs):
            output = self.run_once(input)
            outputs.append(output)
        return np.array(outputs)
    
    def update_network(self, config, weights_normal_scale, biases_normal_scale):
        """
        Update all layers in the network.
        
        Args:
            config (Config): Configuration object
            weights_normal_scale (float): Scale factor for weight updates
            biases_normal_scale (float): Scale factor for bias updates
            
        Returns:
            Network: Updated network (self)
        """
        for layer_num, layer in enumerate(self.layers):
            layer.weights = layer.update_weights(config, weights_normal_scale)
            layer.biases = layer.update_biases(config, biases_normal_scale)
        return self
    
    def get_correct_output(self, inputs):
        """
        Calculate the correct output for given inputs.
        
        Args:
            inputs (numpy.ndarray): Input array
            
        Returns:
            numpy.ndarray: Correct output vector
            
        Raises:
            AttributeError: If input boolean is not 0 or 1
        """
        if inputs[-1] == 0:
            delta = np.zeros_like(inputs[0:4])
            x_arg_max = np.argmax(inputs[0:4])
        elif inputs[-1] == 1:
            delta = np.zeros_like(inputs[4:8])
            x_arg_max = np.argmax(inputs[4:8])
        else:
            raise AttributeError('Got bool not equal to 1,0)')
        delta[x_arg_max] = 1
        return delta
    
class ActivationFunctions:
    """
    Collection of activation functions for neural networks.
    
    This class provides various activation functions including sigmoid,
    ReLU, linear, and softmax with optional exponent capping.
    """
    
    def __init__(self, config):
        """
        Initialize activation functions with configuration.
        
        Args:
            config (Config): Configuration object with exponent cap settings
        """
        self.activation_functions = {
            'sigmoid' : self.sigmoid,
            'relu' : self.relu,
            'linear' : self.linear,
            'softmax' : self.softmax
        }

        self.exp_cap = config.exp_cap

        return None

    def sigmoid(self, x:float):
        """
        Sigmoid activation function.
        
        Args:
            x (float): Input value
            
        Returns:
            float: Sigmoid output
        """
        # if x < -1*self.exp_cap:
        #     x = -1*self.exp_cap
        # elif x > self.exp_cap:
        #     x = self.exp_cap
        z = 1/(1 + np.exp(-x))
        # if np.isnan(z):
        #     if x > 0:
        #         return np.float64(0)
        #     else:
        #         return np.float64(1)
        return z

    def relu(self, x:float):
        """
        Rectified Linear Unit (ReLU) activation function.
        
        Args:
            x (float): Input value
            
        Returns:
            float: ReLU output
        """
        z = np.max([0,x])
        return z

    def linear(self, x:float):
        """
        Linear activation function (identity).
        
        Args:
            x (float): Input value
            
        Returns:
            float: Linear output (same as input)
        """
        return x

    def softmax(self, a:np.array):
        """
        Softmax activation function.
        
        Args:
            a (numpy.ndarray): Input array
            
        Returns:
            numpy.ndarray: Softmax output
        """
        # for i, ele in enumerate(a):
        #     if ele > self.exp_cap:
        #         a[i] = self.exp_cap
        #     elif ele < -1*self.exp_cap:
        #         a[i] = -1*self.exp_cap
        z = np.exp(a)/(np.exp(a).sum())  
        return z
        

class RewardFunctions:
    """
    Collection of reward functions for training neural networks.
    
    This class provides various reward functions for evaluating
    network performance including cosine similarity, Manhattan distance,
    and percentage correct outputs.
    """

    def __init__(self, config):
        """
        Initialize reward functions with configuration.
        
        Args:
            config (Config): Configuration object with reward function settings
        """
        self.reward_functions = {
            'cosine_sim_reward': self.cosine_sim_reward,
            'manhattan_distance_reward': self.manhattan_distance_reward,
            'percentage_correct_reward' : self.percent_correct_outputs_reward
        }
        if config.reward_function is not None:
            self.reward_function = self.reward_functions[config.reward_function]

        return None
    
    def percent_correct_outputs_reward(self, output, y):
        """
        Calculate reward based on percentage of correct outputs.
        
        Args:
            output (numpy.ndarray): Network output
            y (numpy.ndarray): Target output
            
        Returns:
            bool: True if output matches target, False otherwise
        """
        # self.make_reward_assertions(x, y)
        
        delta = np.zeros_like(output)
        x_arg_max = np.argmax(output)
        delta[x_arg_max] = 1
        # delta = np.zeros_like(x)
        # x_arg_max = np.argmax(x)
        # delta[x_arg_max] = 1
        return (delta == y).all()
        

    def cosine_sim_reward(self, x, y):
        """
        Calculate reward based on cosine similarity.
        
        Args:
            x (numpy.ndarray): Network output
            y (numpy.ndarray): Target output
            
        Returns:
            float: Cosine similarity reward
        """
        self.make_reward_assertions(x, y)
        return 1.0-distance.cosine(x, y)
    
    def manhattan_distance_reward(self, x, y):
        """
        Calculate reward based on Manhattan distance.
        
        Args:
            x (numpy.ndarray): Network output
            y (numpy.ndarray): Target output
            
        Returns:
            float: Manhattan distance reward
        """
        self.make_reward_assertions(x,y)
        return (1.0 - (np.abs(x-y).sum())/x.shape[0])
    
    def make_reward_assertions(self, x, y):
        """
        Make assertions about input arrays for reward calculations.
        
        Args:
            x (numpy.ndarray): First array
            y (numpy.ndarray): Second array
            
        Raises:
            AssertionError: If arrays don't meet requirements
        """
        assert ((x.max() <= 1.0) and (x.min() >= 0.0))
        assert ((y.max() <= 1.0) and (y.min() >= 0.0))
        assert (len(x.shape) == 1) and (len(y.shape) == 1)
        assert (x.shape == y.shape)
        return None
    
def make_input_row(config):
    """
    Generate a single input row for training.
    
    Creates random positions for bee, food, queen, and hornet,
    ensuring no two objects occupy the same position.
    
    Args:
        config (Config): Configuration object with board size
        
    Returns:
        tuple: (regular_input_row, hornet_input_row) arrays
    """
    board_size = config.board_size 
    position1 = np.random.choice(range(board_size), 2)/board_size #bee
    position2 = np.random.choice(range(board_size), 2)/board_size #food
    position3 = np.random.choice(range(board_size), 2)/board_size #queen
    position4 = np.random.choice(range(board_size), 2)/board_size #hornet
    has_food = np.expand_dims(np.random.choice([0,1]),0)

    regular_input_row = np.concatenate((position1, position2, position3, has_food))
    hornet_input_row = np.concatenate((position1, position4))

    if (position1 == position2).all() or \
        (position1 == position3).all() or \
        (position1 == position4).all() or \
        (position2 == position3).all() or \
        (position2 == position4).all() or \
        (position3 == position4).all():
        regular_input_row, hornet_input_row = make_input_row(config)

    return np.array(regular_input_row), np.array(hornet_input_row)

def make_inputs(config):
    """
    Generate multiple input rows for training.
    
    Args:
        config (Config): Configuration object with number of inputs
        
    Returns:
        tuple: (regular_inputs, hornet_inputs) arrays
    """
    size = config.num_inputs
    regular_inputs = []
    hornet_inputs = []
    for _ in range(size):
        regular_input_row, hornet_input_row = make_input_row(config)
        regular_inputs.append(regular_input_row)
        hornet_inputs.append(hornet_input_row)
    
    regular_inputs = np.array(regular_inputs)
    hornet_inputs = np.array(hornet_inputs)

    return np.array(regular_inputs), np.array(hornet_inputs)

def output_choice(hornet_exists, net):
    """
    Make a choice based on hornet existence and network output.
    
    Args:
        hornet_exists (bool): Whether a hornet exists
        net (Network): Neural network
        
    Returns:
        float: Network output value
    """
    output = np.matmul(hornet_exists, net.weights) + net.bias
    return output[0][0]

def choose_output(chosen_output):
    """
    Print the chosen output type.
    
    Args:
        chosen_output (float): Network output value
    """
    if chosen_output > 0:
        print('hornet_output')
    else:
        print('regular_outputs')

def run(moral_config, viz=True):
    """
    Run the moral evolution experiment.
    
    This function implements the main evolutionary algorithm for
    training moral decision networks through gameplay.
    
    Args:
        moral_config (Config): Configuration for moral network training
    """
    networks = {}
    alive_networks = {}


    # Init networks
    print('Initializing TF-trained networks')
    regular_network = tf.keras.models.load_model(os.path.join('keras_models', 'regular_model.keras'))
    hornet_network = tf.keras.models.load_model(os.path.join('keras_models', 'hornet_model.keras'))

    print('Initializing morality networks...')
    for net_id in range(moral_config.num_nets):
        networks[net_id] = [Network(moral_config), True, 0]

    # Loop until stopping criteria
    while len(list(alive_networks.keys())) < moral_config.num_nets:
        print(alive_networks)
        print('Playing Games...')
        max_scores = {}
        
        for net_id in networks.keys():
            moral_net = networks[net_id][0]
            print('Now playing Game: ', net_id)
            game = Game()
            queen_alive, max_score = play_game(game, moral_net, regular_network, hornet_network, viz=viz)
            max_scores[net_id] = max_score
            networks[net_id] = [moral_net, queen_alive, max_score]
            max_id = net_id
        
        print('Destroying networks with dead queen.')
        for net_id in networks.keys():
            if networks[net_id][1]:
                alive_networks[net_id] = networks[net_id]
                if len(list(alive_networks.keys())) == moral_config.num_nets:
                    break
            else:
                del max_scores[net_id]

        print('Destroying Networks with low performance.')
        if len(list(max_scores.keys())) > 0:
            cutoff_reward = np.percentile(list(max_scores.values()), moral_config.reward_percentile)
            for key in max_scores.keys():
                if (max_scores[key] < cutoff_reward) and (key in alive_networks.keys()):
                    del alive_networks[key]

        num_to_create = moral_config.num_nets - len(list(alive_networks.keys()))
        print('number to create: ', num_to_create)
        networks = alive_networks.copy()
        print(networks is alive_networks)
        for _ in range(num_to_create):
            if len(alive_networks.keys()) > 0:
                max_id += 1
                net_id = np.random.choice(list(alive_networks.keys()))
                networks[max_id] = [networks[net_id][0].update_network(moral_config, moral_config.update_weights_normal_scale, moral_config.update_biases_normal_scale), True, 0]
            else:
                max_id += 1
                net = Network(moral_config)
                networks[max_id] = [net, True, 0]

        weights = []
        for value in networks.values():
            net = value[0]
            for layer in net.layers:
                weight = layer.weights[0]
                weights.append(weight[0])
        avg_weight = np.mean(weights)

    print(avg_weight)

    return None

def ensure_best(net):
    """
    Ensure the network matches the best saved weights and biases.
    
    Args:
        net (Network): Network to check
        
    Returns:
        tuple: (bool, Network) - Whether network matches best, and the network
    """
    bw_equal = []
    for i, layer in enumerate(net.layers):
        biases = np.loadtxt(f"best_biases_{i}.csv", delimiter=",", dtype=float)
        bw_equal.append((biases == layer.biases).all())
        weights = np.loadtxt(f"best_weights_{i}.csv", delimiter=",", dtype=float)
        bw_equal.append((weights == layer.weights).all())
    bw_equal = np.array(bw_equal)
    return bw_equal.all(), net

def get_correct_output(inputs):
    """
    Calculate the correct output for given inputs.
    
    Args:
        inputs (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: Correct output vector
        
    Raises:
        AttributeError: If input boolean is not 0 or 1
    """
    if inputs[-1] == 0:
        delta = np.zeros_like(inputs[0:4])
        x_arg_max = np.argmax(inputs[0:4])
    elif inputs[-1] == 1:
        delta = np.zeros_like(inputs[4:8])
        x_arg_max = np.argmax(inputs[4:8])
    else:
        raise AttributeError('Got bool not equal to 1,0)')
    delta[x_arg_max] = 1
    return delta

def test(config, inputs, net):
    """
    Test network performance on in-sample and out-of-sample data.
    
    Args:
        config (Config): Configuration object
        inputs (numpy.ndarray): Input data
        net (Network): Network to test
    """
    best_net = net
    regular_network = Network(regular_config)
    hornet_network = Network(hornet_config)
    rfs = RewardFunctions(config)
    print('IS Test:')
    output_correct = []
    outputs = []
    correct_outputs = []
    rfs_outputs = []
    for input in inputs:
        output = best_net.run_once(input)
        delta = np.zeros_like(output)
        x_arg_max = np.argmax(output)
        delta[x_arg_max] = 1
        correct_output = get_correct_output(input)
        output_correct.append((delta == correct_output).all())
        rfs_output = rfs.percent_correct_outputs_reward(output, correct_output)
        rfs_outputs.append(rfs_output)

    print(np.mean(rfs_outputs))
    regular_inputs, hornet_inputs = make_inputs(config)
    regular_outputs = regular_network.get_outputs(regular_inputs)
    hornet_outputs = hornet_network.get_outputs(hornet_inputs)
    final_inputs = []
    for row in np.append(regular_outputs, hornet_outputs,1):
        final_inputs.append(np.append(row, np.random.choice([0,1])))
    new_inputs = np.array(final_inputs)
    print('OOS Test:')
    oos_output_correct = []
    for input in new_inputs:
        outputs = best_net.run_once(input)
        delta = np.zeros_like(outputs)
        x_arg_max = np.argmax(outputs)
        delta[x_arg_max] = 1
        correct_outputs = get_correct_output(input)
        #print(delta, correct_outputs, (delta == correct_outputs).all())
        oos_output_correct.append((delta == correct_outputs).all())
    print(np.mean(oos_output_correct))
    return None


def demo(moral_net, viz=True):
    """
    Run a demonstration of the moral network.
    
    Args:
        moral_net (Network): Moral network to demonstrate
    """
    game = Game()
    regular_network = tf.keras.models.load_model(os.path.join('keras_models', 'regular_model.keras'))
    hornet_network = tf.keras.models.load_model(os.path.join('keras_models', 'hornet_model.keras'))
    queen_alive, score = play_game(game, moral_net, regular_network, hornet_network, viz=viz)
    print({'queen_alive': queen_alive, 'score': score})
    return queen_alive, score

def check_outputs(regular_config):
    """
    Check outputs between Keras model and custom network implementation.
    
    Args:
        regular_config (Config): Configuration for regular network
    """
    keras_regular_model = tf.keras.models.load_model('regular_model.keras')
    regular_network = Network(regular_config)

    regular_inputs, hornet_inputs = make_inputs(regular_config)
    keras_outputs = keras_regular_model.predict(regular_inputs)
    my_outputs = regular_network.get_outputs(regular_inputs)

    keras_choice = [np.argmax(x) for x in np.array(keras_outputs)]
    my_choice = [np.argmax(x) for x in np.array(my_outputs)]
    print(list(zip(keras_choice, my_choice)))
    print(keras_choice == my_choice)

if __name__ == '__main__':
    write_se_config()
    moral_config_file = os.path.join('config_files', 'morality_layer_config.json')
    hornet_config_file = os.path.join('config_files', 'hornet_network_config.json')
    regular_config_file = os.path.join('config_files', 'regular_network_config.json')
    moral_config = Config(config_file=moral_config_file)
    hornet_config = Config(config_file=hornet_config_file)
    regular_config = Config(config_file=regular_config_file)

    run(moral_config)




