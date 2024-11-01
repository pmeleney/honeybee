class GameState:
    #Gameplay Vars
    MAX_TURNS = 500

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
    NUM_FLOWERS = 150
    FLOWER_REFRESH_TURNS = None
    BLANK_MOAT = (1,1)
    BLANK_RIM = 2

    #Bee Vars
    NUM_STARTING_BEES = 1
    FOOD_PER_BEE = 10

    #Hornet Vars
    HORNETS_EXIST = True
    NUM_STARTING_HORNETS = 0
    HORNET_MOVE_SPEED = 3
    HORNET_FREQUENCY = 35
    HORNET_RANDOM_ELE = 1
    

class GameVars:
    reward = 0.0
    index = 0
    turn_num = 0
    bees_generated = 0
    food_collected = 0
    hornet_created = False
    queen_alive = True