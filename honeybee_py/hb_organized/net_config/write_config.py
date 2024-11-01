import json
import os

def write_se_config():
    morality_layer_config = {
        # Run Attributes
        'reward_percentile' : 0,
        #'board_size' : 30,
        'reward_function' : None,

        # Net Attributes
        'net_shape' : [[1,1]],
        'layer_activations' : ['softmax'],
        'num_nets' : 2,

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

    with open(os.path.join('net_config', 'morality_layer_config.json'), 'w') as f:
        json.dump(morality_layer_config, f)

    return None

if __name__ == '__main__':
    write_se_config()