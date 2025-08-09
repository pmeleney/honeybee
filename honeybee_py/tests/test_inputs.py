import unittest
import numpy as np

from honeybee_py.game.game import Game
from honeybee_py.game.gameobjects import Hornet


class TestInputEncoders(unittest.TestCase):
    def test_regular_inputs_new_shape_and_range(self):
        g = Game()
        bee = g.bees[0]
        flower = g.flowers[0]
        x = g.get_regular_inputs_new(bee, flower, g.queen)
        self.assertEqual(x.shape, (1, 5))
        self.assertTrue(np.issubdtype(x.dtype, np.floating))
        self.assertTrue(np.all(x[:, :4] <= 1.0) and np.all(x[:, :4] >= -1.0))
        self.assertIn(float(bee.has_food), [0.0, 1.0])

    def test_hornet_inputs_new_inv_exists_flag(self):
        g = Game()
        bee = g.bees[0]
        # Case: no hornets
        g.hornets = []
        hx, hy = g.queen.position[0]
        x = g.get_hornet_inputs_new(bee, np.array([hx, hy]))
        self.assertEqual(x.shape, (1, 5))
        inv_exists = float(x[0, 4])
        self.assertEqual(inv_exists, 1.0)

        # Case: hornet present
        g.hornets = [Hornet(position=(0, 0))]
        x2 = g.get_hornet_inputs_new(bee, np.array([hx, hy]))
        inv_exists2 = float(x2[0, 4])
        self.assertEqual(inv_exists2, 0.0)


if __name__ == '__main__':
    unittest.main()


