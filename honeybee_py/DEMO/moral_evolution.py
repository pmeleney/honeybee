import numpy as np
rng = np.random.default_rng()
from game.game import Game, play_game
import json
import os
from scipy.spatial import distance
import json
import tensorflow as tf

class Config():
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f) # To Do

            self.name = 'morality_layer'
            self.run_load = 'run'
            self.reward_function = config['reward_function']
            self.reward_percentile = config['reward_percentile']
            #self.board_size = config['board_size']
            
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

            return None

class Layer:

    def __init__(self, config, layer_num):
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
        Initializes weights of Layer using vars described in config.
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
        Initializes biases of Layer using vars described in config.
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
        Update weights according to config.
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
        Update biases according to config.
        """
        if config.update_biases_dist == 'uniform':
            update_to_biases = config.update_biases_uniform_range * ((2*np.random.random(self.layer_shape[1])) - 1)
        elif config.update_biases_dist == 'normal':
            update_to_biases = rng.normal(loc=config.update_biases_normal_loc, scale=biases_normal_scale, size=[self.layer_shape[1]])
        return self.biases + update_to_biases
    
    def load_weights(self, layer_num, config):
        layer_weights = np.loadtxt(config.weights_files[layer_num], delimiter=',', dtype=float) 
        return layer_weights

    def load_biases(self, layer_num, config):
        layer_biases = np.loadtxt(config.biases_files[layer_num], delimiter=',', dtype=float) 
        return layer_biases
    
    def copy(self, config, layer_num):
        new_layer = Layer(config, layer_num)
        new_layer.weights = self.weights.copy()
        new_layer.biases = self.biases.copy()
        return new_layer

class Network:

    def __init__(self, config):
        self.layers = []
        for layer_num in range(len(config.net_shape)):
            layer = Layer(config, layer_num)
            self.layers.append(layer)
        self.reward_function = self.select_reward_function(config)
        return None
    
    def copy(self, config):
        new_net = Network(config)
        for layer_num, layer in enumerate(self.layers):
            new_net.layer = layer.copy(config, layer_num)
        return new_net
    
    def select_reward_function(self, config):
        if config.reward_function is not None:
            rfs = RewardFunctions(config)
            return rfs.reward_function
        else:
            return None
    
    def run_once(self, input):
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
        reward = 0
        for _, input in enumerate(inputs):
            output = self.run_once(input)
            correct_output = self.get_correct_output(input)
            reward += self.reward_function(output, correct_output)
            if display_results:
                print(output, correct_output, reward)
        return reward/(inputs.shape[0])
    
    def get_outputs(self, inputs, display_results=False):
        outputs = []
        for _, input in enumerate(inputs):
            output = self.run_once(input)
            outputs.append(output)
        return np.array(outputs)
    
    def update_network(self, config, weights_normal_scale, biases_normal_scale):
        for layer_num, layer in enumerate(self.layers):
            layer.weights = layer.update_weights(config, weights_normal_scale)
            layer.biases = layer.update_biases(config, biases_normal_scale)
        return self
    
    def get_correct_output(self, inputs):
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
    
    def __init__(self, config):
        self.activation_functions = {
            'sigmoid' : self.sigmoid,
            'relu' : self.relu,
            'linear' : self.linear,
            'softmax' : self.softmax
        }

        self.exp_cap = config.exp_cap

        return None

    def sigmoid(self, x:float):
        if x < -1*self.exp_cap:
            x = -1*self.exp_cap
        elif x > self.exp_cap:
            x = self.exp_cap
        z = 2/(1 + np.exp(-x)) - 1
        # if np.isnan(z):
        #     if x > 0:
        #         return np.float64(0)
        #     else:
        #         return np.float64(1)
        return z

    def relu(self, x:float):
        z = np.max([0,x])
        return z

    def linear(self, x:float):
        return x

    def softmax(self, a:np.array):
        for i, ele in enumerate(a):
            if ele > self.exp_cap:
                a[i] = self.exp_cap
            elif ele < -1*self.exp_cap:
                a[i] = -1*self.exp_cap
        z = np.exp(a)/(np.exp(a).sum())  
        return z
        

class RewardFunctions:

    def __init__(self, config):
        self.reward_functions = {
            'cosine_sim_reward': self.cosine_sim_reward,
            'manhattan_distance_reward': self.manhattan_distance_reward,
            'percentage_correct_reward' : self.percent_correct_outputs_reward
        }
        if config.reward_function is not None:
            self.reward_function = self.reward_functions[config.reward_function]

        return None
    
    def percent_correct_outputs_reward(self, output, y):
        # self.make_reward_assertions(x, y)
        
        delta = np.zeros_like(output)
        x_arg_max = np.argmax(output)
        delta[x_arg_max] = 1
        # delta = np.zeros_like(x)
        # x_arg_max = np.argmax(x)
        # delta[x_arg_max] = 1
        return (delta == y).all()
        

    def cosine_sim_reward(self, x, y):
        self.make_reward_assertions(x, y)
        return 1.0-distance.cosine(x, y)
    
    def manhattan_distance_reward(self, x, y):
        self.make_reward_assertions(x,y)
        return (1.0 - (np.abs(x-y).sum())/x.shape[0])
    
    def make_reward_assertions(self, x, y):
        assert ((x.max() <= 1.0) and (x.min() >= 0.0))
        assert ((y.max() <= 1.0) and (y.min() >= 0.0))
        assert (len(x.shape) == 1) and (len(y.shape) == 1)
        assert (x.shape == y.shape)
        return None

def run(moral_config):
    progression = []
    networks = {}
    alive_networks = {}

    # Init networks
    print('Initializing TF-trained networks')
    qf_model = tf.keras.models.load_model(os.path.join('tf_models', 'qf_model.keras'))
    h_model = tf.keras.models.load_model(os.path.join('tf_models', 'h_model.keras'))

    print('Initializing morality networks...')
    for net_id in range(moral_config.num_nets):
        networks[net_id] = [Network(moral_config), True, 0]

    # Loop until stopping criteria
    run = 0
    nets_played = []
    while len(list(alive_networks.keys())) < moral_config.num_nets:
        print(alive_networks)
        print('Playing Games...')
        max_scores = {}
        
        for net_id in networks.keys():
            if net_id not in nets_played:
                moral_net = networks[net_id][0]
                print('Now playing Game: ', net_id)
                game = Game()
                queen_alive, max_score = play_game(game, moral_net, qf_model, h_model, viz=False)
                nets_played.append(net_id)
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

        progression.append([run, avg_weight])
        run+=1

    max_score = -9999
    for net in networks.values():
        if net[2] > max_score:
            best_net = net[0]
            max_score = net[2]

    print(best_net.layers[0].weights)
    print(best_net.layers[0].biases)
    print(max_score)

    return best_net, progression

def demo(moral_net):
    game = Game()
    qf_model = tf.keras.models.load_model(os.path.join('tf_models', 'qf_model.keras'))
    h_model = tf.keras.models.load_model(os.path.join('tf_models', 'h_model.keras'))
    play_game(game, moral_net, qf_model, h_model, viz=True)
    return None

if __name__ == '__main__':
    moral_config_file = os.path.join('net_config', 'morality_layer_config.json')
    moral_config = Config(config_file=moral_config_file)

    if moral_config.run_load == 'run':
        moral_net, progression = run(moral_config)
        for i, layer in enumerate(moral_net.layers):
            weights = layer.weights
            np.savetxt(os.path.join('morality_layer', f"weights_{i}.csv"), weights, delimiter=",")
            biases = layer.biases
            np.savetxt(os.path.join('morality_layer', f"biases_{i}.csv"), biases, delimiter=",")

    elif moral_config.run_load == 'load':
        moral_net = Network(moral_config)
        for i, layer in enumerate(moral_net.layers):
            weights = np.loadtxt(os.path.join('morality_layer', f"weights_{i}.csv"), dtype=float, delimiter=',')
            moral_net.layers[i].weights = weights
            biases = np.loadtxt(os.path.join('morality_layer', f"biases_{i}.csv"), dtype=float, delimiter=',')
            moral_net.layers[i].biases = biases
    
    else:
        raise AttributeError('config.run_load should be "run" or "load"')
    
    demo(moral_net)
    print('DONE.')




