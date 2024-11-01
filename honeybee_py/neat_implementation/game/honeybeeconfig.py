

class GameState:
    #Gameplay Vars
    MAX_TURNS = 50

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
    FLOWER_REFRESH_TURNS = 5
    BLANK_MOAT = (1,1)
    BLANK_RIM = 2

    #Bee Vars
    NUM_STARTING_BEES = 1

    #Hornet Vars
    HORNETS_EXIST = False
    NUM_STARTING_HORNETS = 0
    HORNET_START_TRUNS = 0
    HORNET_FREQUENCY = 5
    HORNET_RANDOM_ELE = 1
    

class GameVars:
    reward = 0.0
    index = 0
    turn_num = 0
    bees_generated = 0
    food_collected = 0
    hornet_created = False
    queen_alive = True