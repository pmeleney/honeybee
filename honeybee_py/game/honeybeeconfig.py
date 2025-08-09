class GameState:
    """
    Configuration class containing game state parameters and settings.
    
    This class defines all the static configuration parameters for the honeybee game,
    including board dimensions, queen settings, flower parameters, bee settings,
    and hornet behavior.
    """
    #Gameplay Vars
    MAX_TURNS = 124

    #Gameboard Vars
    GAME_BOARD_SHAPE = 'square'
    if GAME_BOARD_SHAPE == 'square':
        NUM_GRID = (30,30)
    else:
        raise AttributeError('GAME_BOARD_SHAPE must be square.')
    
    #Queen Vars
    QUEEN_POSITION = 'center'
    QUEEN_SIZE = (2,2)

    #Flower Vars
    NUM_FLOWERS = 256
    FLOWER_REFRESH_TURNS = 32
    BLANK_MOAT = (1,1)
    BLANK_RIM = 2

    #Bee Vars
    NUM_STARTING_BEES = 2

    #Hornet Vars
    HORNETS_EXIST = True
    NUM_STARTING_HORNETS = 0
    HORNET_START_TRUNS = 0
    HORNET_FREQUENCY = 25
    HORNET_RANDOM_ELE = 1
    

from dataclasses import dataclass


@dataclass
class GameVars:
    """
    Dynamic game variables that change during gameplay.
    
    Tracks cumulative reward, step index, turn number, hive stats, and events.
    """
    reward: float = 0.0
    index: int = 0
    turn_num: int = 0
    bees_generated: int = 0
    food_collected: int = 0
    hornet_created: bool = False
    queen_alive: bool = True
    hornets_killed: int = 0