import numpy as np

from game.helpers import _fill_rect, _flatten_list, _distance
from game.honeybeeconfig import GameState, GameVars


game_state = GameState()

# np.random.seed = 47

class GameObject:

    def __init__(self):
        self.name = None
        self.color = (0,0,0)
        self.position = [0,0]

class Bee(GameObject):

    def __init__(self):
        self.name = 'Honeybee'
        self.color = (255,165,0)
        self.has_food = False
        self.position = [np.random.choice(range(int(np.floor(0.25*(game_state.NUM_GRID[0]-1))), (int(np.floor(0.75*(game_state.NUM_GRID[0]-1)))))), np.random.choice(range(int(np.floor(0.25*(game_state.NUM_GRID[1]-1))), (int(np.floor(0.75*(game_state.NUM_GRID[1]-1))))))]
    
    def check_overlap(self, queen, flowers, hornets):
        game_objects = list(([queen], flowers))
        game_objects = _flatten_list(game_objects)
        for game_object in game_objects:
            if game_object.name == 'Queen':
                if tuple(self.position) in (game_object.position):
                    return 'Queen'
            else:
                if list(self.position) == list(game_object.position):
                    return game_object.name
        return None
    
    def find_nearest_flower(self, flowers):
        flowers_with_food = [flower for flower in flowers if flower.has_food]
        min_distance = 9999
        for _, flower in enumerate(flowers_with_food):
            bee_pos = self.position
            flower_pos = flower.position
            flower_distance = _distance(bee_pos, flower_pos)
            if flower_distance < min_distance:
                min_distance = flower_distance
                nearest_flower = flower
        nearest_flower.color = [255,255,255]
        return nearest_flower

    def get_food(self, flower):
        if flower.has_food: #and np.logical_not(self.has_food):
            self.has_food = True
            flower.has_food = False #change later
            flower.color = flower.nofood_color
        return None
    
    def drop_food(self):
        if self.has_food:
            self.has_food = False
        return None
    
class Queen(GameObject):

    def __init__(self, size, position):
        self.name = "Queen"
        self.size = size
        self.position = position
        self.color = (128,0,128)

class Flower(GameObject):

    def __init__(self, position):
        self.name = 'Flower'
        self.color = (7, 218, 99)
        self.nofood_color = (1,50,32)
        self.has_food = True
        self.position = position
        self.food_taken_on = 0

class Hornet(GameObject):
    
    def __init__(self):
        self.position = (0,0)
        self.color = (255,0,0)

    def check_overlap(self, queen):
        if tuple(self.position) in (queen.position):
            return 'Queen'
        return None

