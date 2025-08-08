import numpy as np

from .helpers import _fill_rect, _flatten_list, _distance
from .honeybeeconfig import GameState, GameVars


game_state = GameState()

class GameObject:
    """
    Base class for all game objects in the honeybee simulation.
    
    Provides common attributes and methods for bees, flowers, hornets, and the queen.
    """

    def __init__(self):
        """
        Initialize a game object with default attributes.
        """
        self.name = None
        self.color = (0,0,0)
        self.position = [0,0]

class Bee(GameObject):
    """
    Represents a honeybee in the simulation.
    
    Bees can collect food from flowers and return it to the queen.
    They can also fight hornets when they have food.
    """

    def __init__(self):
        """
        Initialize a bee with random position and default attributes.
        """
        self.name = 'Honeybee'
        self.score = 0
        self.color = (255,165,0)
        self.has_food = False
        self.position = [np.random.choice(range(int(np.floor(0.25*(game_state.NUM_GRID[0]-1))), (int(np.floor(0.75*(game_state.NUM_GRID[0]-1)))))), np.random.choice(range(int(np.floor(0.25*(game_state.NUM_GRID[1]-1))), (int(np.floor(0.75*(game_state.NUM_GRID[1]-1))))))]
    
    def check_overlap(self, queen, flowers, hornets):
        """
        Check if the bee overlaps with any game objects.
        
        Args:
            queen (Queen): The queen bee object
            flowers (list): List of flower objects
            hornets (list): List of hornet objects
            
        Returns:
            str or None: The name of the overlapped object ('Queen', 'Hornet', 'Flower') or None if no overlap
        """
        game_objects = list(([queen], flowers, hornets))
        game_objects = _flatten_list(game_objects)
        for game_object in game_objects:
            if game_object.name == 'Queen':
                if tuple(self.position) in (game_object.position):
                    return 'Queen'
            if game_object.name == 'Hornet':
                if list(self.position) == list(game_object.position):
                    return 'Hornet'
            else:
                if list(self.position) == list(game_object.position):
                    return game_object.name
        return None
    
    def find_nearest_flower_with_food(self, flowers):
        """
        Find the nearest flower that still has food available.
        
        Args:
            flowers (list): List of flower objects to search through
            
        Returns:
            Flower: The nearest flower with food, or None if no flowers have food
        """
        flowers_with_food = [flower for flower in flowers if flower.has_food]
        min_distance = 9999
        for _, flower in enumerate(flowers_with_food):
            flower.color = (7, 218, 99)
            bee_pos = self.position
            flower_pos = flower.position
            flower_distance = _distance(bee_pos, flower_pos)
            if flower_distance < min_distance:
                min_distance = flower_distance
                nearest_flower = flower
        nearest_flower.color = [255,255,255]
        return nearest_flower

    def get_food(self, flower):
        """
        Collect food from a flower if the bee doesn't already have food.
        
        Args:
            flower (Flower): The flower to collect food from
            
        Returns:
            None
        """
        if flower.has_food and np.logical_not(self.has_food):
            self.has_food = True
            flower.has_food = False #change later
            flower.color = flower.nofood_color
        return None
    
    def drop_food(self):
        """
        Drop food at the queen's location.
        
        Returns:
            None
        """
        if self.has_food:
            self.has_food = False
        return None
    
class Queen(GameObject):
    """
    Represents the queen bee in the simulation.
    
    The queen is the central figure that bees return food to.
    She occupies a 2x2 area in the center of the board.
    """

    def __init__(self, size, position):
        """
        Initialize the queen with size and position.
        
        Args:
            size (tuple): Size of the queen as (width, height)
            position (list): List of coordinate tuples representing queen's position
        """
        self.name = "Queen"
        self.size = size
        self.position = position
        self.color = (128,0,128)

class Flower(GameObject):
    """
    Represents a flower that provides food for bees.
    
    Flowers start with food available and change color when food is collected.
    """

    def __init__(self, position):
        """
        Initialize a flower at the specified position.
        
        Args:
            position (tuple): (x, y) coordinates for the flower's position
        """
        self.name = 'Flower'
        self.color = (7, 218, 99)
        self.nofood_color = (1,50,32)
        self.has_food = True
        self.position = position
        self.food_taken_on = 0

class Hornet(GameObject):
    """
    Represents a hornet that threatens the queen and bees.
    
    Hornets move towards the queen and can kill bees or the queen on contact.
    """
    
    def __init__(self, position):
        """
        Initialize a hornet at the specified position.
        
        Args:
            position (tuple): (x, y) coordinates for the hornet's position
        """
        self.name = 'Hornet'
        self.position = np.array(position)
        self.color = (255,0,0)

    def check_overlap(self, queen):
        """
        Check if the hornet overlaps with the queen.
        
        Args:
            queen (Queen): The queen bee object
            
        Returns:
            bool: True if hornet overlaps with queen, False otherwise
        """
        if tuple(self.position) in (queen.position):
            return True
        return False

