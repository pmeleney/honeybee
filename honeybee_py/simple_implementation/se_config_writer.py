import json



def write_se_config():
    config = {
        # Run Attributes
        'num_iter': 1,
        'reward_percentile' : 80,
        'reward_function' : 'percent_correct_outputs_reward',
        'stopping_criteria' : 0.75,
        'init_new' : True,
        'board_size' : 1000,
        'num_inputs' : 1000,
        'max_num_iter_without_improvement' : 8,

        # Net Attributes
        'net_shape' : [[4,10], [10,10], [10,10], [10,4]],
        'layer_activations' : ['relu', 'sigmoid', 'sigmoid', 'softmax'],
        'num_nets' : 1000,

        # Exponent Capping
        'exp_cap' : 100,

        # Weights Attributes
            # Init        
        'init_weights_dist' : 'uniform',
        'init_weights_uniform_range' : 30.0,
        'init_weights_normal_loc' : 0.0,
        'init_weights_normal_scale' : 30.0,

            # Update        
        'update_weights_dist' : 'normal',
        'update_weights_uniform_range' : 2.0,
        'update_weights_normal_loc' : 0.0,
        'update_weights_normal_scale' : [4.0, 2.0, 1.0, 0.5, 0.1],

        # Biases Attributes
            # Init
        'init_biases_dist' : 'uniform',
        'init_biases_uniform_range' : 30.0,
        'init_biases_normal_loc' : 0.0,
        'init_biases_normal_scale' : 30.0,
        
            # Update
        'update_biases_dist' : 'normal',
        'update_biases_uniform_range' : 2.0,
        'update_biases_normal_loc' : 0.0,
        'update_biases_normal_scale' : [4.0, 2.0, 1.0, 0.5, 0.1],
    }

    with open('se_config.json', 'w') as f:
        json.dump(config, f)

    return None

if __name__ == '__main__':
    write_se_config()