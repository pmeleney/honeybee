# TODO:
# - Enable checkpoint restoration
# - SGD 

import numpy as np
import json
import sys
from scipy.spatial import distance

class Config():

    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f) # To Do

        # Run Attributes
        self.num_iter = config['num_iter']
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

        return None

class Layer:

    def __init__(self, config, layer_num):
        afs = ActivationFunctions(config)

        self.layer_shape = config.net_shape[layer_num]
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
            layer_weights = ((np.random.normal(loc=config.init_weights_normal_loc, scale=config.init_weights_normal_scale, size=[self.layer_shape[0], self.layer_shape[1]])))
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
            layer_biases = ((np.random.normal(loc=config.init_biases_normal_loc, scale=config.init_biases_normal_scale, size=[self.layer_shape[0], self.layer_shape[1]])))
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
            update_to_weights = np.random.normal(loc=config.update_weights_normal_loc, scale=weights_normal_scale, size=[self.layer_shape[0], self.layer_shape[1]])
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
            update_to_biases = np.random.normal(loc=config.update_biases_normal_loc, scale=biases_normal_scale, size=[self.layer_shape[1]])
        return self.biases + update_to_biases
    
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
        self.num_iter = config.num_iter
        return None
    
    def get_correct_output(inputs):
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
    
    def copy(self, config):
        new_net = Network(config)
        for layer_num, layer in enumerate(self.layers):
            new_net.layer = layer.copy(config, layer_num)
        return new_net
    
    def select_reward_function(self, config):
        rfs = RewardFunctions(config)
        return rfs.reward_function
    
    def run_once(self, input):
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
    
    def update_network(self, config, weights_normal_scale, biases_normal_scale):
        for layer_num, layer in enumerate(self.layers):
            layer.weights = layer.update_weights(config, weights_normal_scale)
            layer.biases = layer.update_biases(config, biases_normal_scale)
    
    def get_correct_output(self, x1, y1, x2, y2):
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
        return np.array(move_vector)
    
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
        z = 1/(1 + np.exp(-x))
        if np.isnan(z):
            if x > 0:
                return np.float64(0)
            else:
                return np.float64(1)
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
            'percent_correct_outputs_reward' : self.percent_correct_outputs_reward
        }

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
    
def make_inputs(config):
    board_size = config.board_size 
    size = config.num_inputs
    position1 = np.random.choice(range(board_size), (size,2))
    position2 = np.random.choice(range(board_size), (size,2))
    positions = np.append(position1, position2, axis=1)
    if (position1 == position2).all():
        positions = make_inputs(board_size, size)
    return positions

def run(config):

    # Init networks
    print('Initializing networks...')
    networks = {}
    max_id = 0
    max_reward = 0
    num_iter_without_improvement = 0
    last_max_reward = 0
    #rfs = RewardFunctions()
    for net_id in range(config.num_nets):
        networks[net_id] = Network(config)
        max_id = net_id

    # Loop until stopping criteria
    inputs = make_inputs(config) # THIS IS NEW
    while max_reward < config.stopping_criteria:
        # Make new inputs for each run
        
        # Get Rewards
        print('Getting rewards...')
        rewards = {}
        for net_id in networks.keys():
            net = networks[net_id]
            reward = net.run(inputs)
            rewards[net_id] = reward
            if reward > max_reward:
                max_reward = reward
        print(f'Rewards generated: {max_reward}')

        # Save best
        print('Saving best output')
        for net_id in rewards.keys():
            if rewards[net_id] == max_reward:
                best_net_id = net_id
                best_net = networks[best_net_id].copy(config)
                best_layers = best_net.layers
                for layer_num, layer in enumerate(best_layers):
                    best_weights = layer.weights
                    np.savetxt(f"best_weights_{layer_num}.csv", best_weights, delimiter=",")
                    best_biases = layer.biases
                    np.savetxt(f"best_biases_{layer_num}.csv", best_biases, delimiter=",")
        # Destroy non-performant networks
        print('Destroying non-performant networks.')
        cutoff_reward = np.percentile(list(rewards.values()), config.reward_percentile)
        for key in rewards.keys():
            if rewards[key] < cutoff_reward:
                del networks[key]

        # Update performant networks
        print('Updating performant networks')
        biases_normal_scale = config.update_biases_normal_scale[0]
        weights_normal_scale = config.update_weights_normal_scale[0]
        for i in range(len(config.update_weights_normal_scale)):
            if num_iter_without_improvement > i*config.max_num_iter_without_improvement:
                biases_normal_scale = config.update_biases_normal_scale[i]
                weights_normal_scale = config.update_biases_normal_scale[i]

        for net_id in networks.keys():
            if net_id == best_net_id:
                continue
            net = networks[net_id]
            net.update_network(config, weights_normal_scale, biases_normal_scale)
        
        # Initialize new networks by updating existing networks
        num_to_choose = len(rewards.keys()) - len(networks.keys())
        if config.init_new:
            max_id += 1
            networks[max_id] = best_net.copy(config) # Keep Best Net
            best_net.run(inputs)
            print('Initializing new networks.')

            while len(networks) < config.num_nets:
                max_id += 1
                networks[max_id] = best_net.copy(config)
                networks[max_id].update_network(config, weights_normal_scale, biases_normal_scale) #Search around best net
                max_id += 1
                networks[max_id] = Network(config) #init new nets
        else:
            print('Generating new networks.')
            if num_to_choose <= len(networks.keys()):
                rnd_network_ids = np.random.choice(list(networks.keys()), num_to_choose, replace=False)
            else:
                rnd_network_ids = np.random.choice(list(networks.keys()), num_to_choose, replace=True)
            while len(networks) < config.num_nets:
                for rnd_network_id in rnd_network_ids:
                    max_id += 1
                    networks[max_id] = networks[rnd_network_id]
                    networks[max_id].update_network(config)

        if max_reward > last_max_reward:
            last_max_reward = max_reward
            num_iter_without_improvement = 0

        print(f'iterations without improvement: {num_iter_without_improvement}.')
        num_iter_without_improvement += 1
        if num_iter_without_improvement > len(config.update_weights_normal_scale)*config.max_num_iter_without_improvement:
            best_net = networks[best_net_id]
            best_layers = best_net.layers
            for layer_num, layer in enumerate(best_layers):
                best_weights = layer.weights
                np.savetxt(f"best_weights_{layer_num}.csv", best_weights, delimiter=",")
                best_biases = layer.biases
                np.savetxt(f"best_biases_{layer_num}.csv", best_biases, delimiter=",")
            return best_net, inputs
    best_net = networks[best_net_id]
    best_layers = best_net.layers
    for layer_num, layer in enumerate(best_layers):
        best_weights = layer.weights
        np.savetxt(f"best_weights_{layer_num}.csv", best_weights, delimiter=",")
        best_biases = layer.biases
        np.savetxt(f"best_biases_{layer_num}.csv", best_biases, delimiter=",")
    return best_net, inputs 

def ensure_best(net):
    bw_equal = []
    for i, layer in enumerate(net.layers):
        biases = np.loadtxt(f"best_biases_{i}.csv", delimiter=",", dtype=float)
        bw_equal.append((biases == layer.biases).all())
        weights = np.loadtxt(f"best_weights_{i}.csv", delimiter=",", dtype=float)
        bw_equal.append((weights == layer.weights).all())
    bw_equal = np.array(bw_equal)
    return bw_equal.all(), net

def test(config, inputs, net):
    best_net = net
    # best_net = Network(config)
    # for layer in best_net:
    #     layer.weights = layer.load_weights()
    #     layer.biases = layer.load_biases()
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
        correct_output = best_net.get_correct_output(*input)
        output_correct.append((delta == correct_output).all())
        rfs_output = rfs.percent_correct_outputs_reward(output, correct_output)
        rfs_outputs.append(rfs_output)

    print(np.mean(rfs_outputs))
    new_inputs = make_inputs(config)
    print('OOS Test:')
    oos_output_correct = []
    for input in new_inputs:
        outputs = best_net.run_once(input)
        delta = np.zeros_like(outputs)
        x_arg_max = np.argmax(outputs)
        delta[x_arg_max] = 1
        correct_outputs = best_net.get_correct_output(*input)
        #print(delta, correct_outputs, (delta == correct_outputs).all())
        oos_output_correct.append((delta == correct_outputs).all())
    print(np.mean(oos_output_correct))
    return None

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = Config(config_file=config_file)
    net, inputs = run(config)
    is_True, net = ensure_best(net)
    assert is_True, 'Best Network not saved :('
    test(config, inputs, net)


