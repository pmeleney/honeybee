class GameState:
    """
    Configuration class containing game state parameters and settings.
    
    This class defines all the static configuration parameters for the honeybee game,
    including board dimensions, queen settings, flower parameters, bee settings,
    and hornet behavior.
    """
    #Gameplay Vars
    MAX_TURNS = 200

    #Gameboard Vars
    GAME_BOARD_SHAPE = 'square'
    if GAME_BOARD_SHAPE == 'square':
        NUM_GRID = (20,20)
    else:
        raise AttributeError('GAME_BOARD_SHAPE must be square.')
    
    #Queen Vars
    QUEEN_POSITION = 'center'
    QUEEN_SIZE = (2,2)

    #Flower Vars
    NUM_FLOWERS = 50
    FLOWER_REFRESH_TURNS = 100
    BLANK_MOAT = (1,1)
    BLANK_RIM = 2

    #Bee Vars
    NUM_STARTING_BEES = 3

    #Hornet Vars
    HORNETS_EXIST = True
    NUM_STARTING_HORNETS = 0
    HORNET_START_TRUNS = 0
    HORNET_FREQUENCY = 30
    HORNET_RANDOM_ELE = 1
    

class GameVars:
    """
    Dynamic game variables that change during gameplay.
    
    This class tracks the current state of the game including scores,
    turn numbers, and various counters that are updated as the game progresses.
    """
    reward = 0.0
    index = 0
    turn_num = 0
    bees_generated = 0
    food_collected = 0
    hornet_created = False
    queen_alive = True