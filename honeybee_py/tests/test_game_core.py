import unittest
import numpy as np

from honeybee_py.game.game import Game


class TestGameCore(unittest.TestCase):
    def test_fill_board_and_queen(self):
        g = Game()
        self.assertTrue(len(g.queen.position) > 0)
        gb = g.update_game_board()
        # Board shape should be (width, height, 3)
        self.assertEqual(gb.shape[2], 3)

    def test_bee_overlap_and_scoring(self):
        g = Game()
        # Place a bee at a flower, collect, then move to queen and drop
        bee = g.bees[0]
        flower = g.flowers[0]
        bee.position = list(flower.position)
        bee.get_food(flower)
        self.assertTrue(bee.has_food)

        qx, qy = g.queen.position[0]
        bee.position = [qx, qy]
        bee.drop_food()
        self.assertFalse(bee.has_food)

    def test_movement_edges(self):
        g = Game()
        bee = g.bees[0]
        # put bee at top-left corner and try moving up/left; should correct
        bee.position = [0, 0]
        # create outputs preferring 'up'
        outputs = np.array([[1.0, 0.0, 0.0, 0.0]])
        new_pos = g.net_move(bee, outputs)
        self.assertNotEqual(new_pos, [0, -1])


if __name__ == '__main__':
    unittest.main()


