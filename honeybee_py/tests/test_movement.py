import unittest
import numpy as np

from honeybee_py.game.game import Game


class TestMovement(unittest.TestCase):
    def test_get_next_best_move_respects_forbidden(self):
        g = Game()
        # Prefer 'up', but forbid it
        norm = np.array([[0.9, 0.05, 0.03, 0.02]], dtype=np.float32)
        move = g.get_next_best_move(norm, ['up'])
        self.assertNotEqual(move, 'up')
        self.assertIn(move, ['dn', 'rt', 'lt'])

    def test_corner_correction_top_left(self):
        g = Game()
        bee = g.bees[0]
        bee.position = [0, 0]
        # Output strongly prefers 'up', which is forbidden; should redirect
        outputs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        new_pos = g.net_move(bee, outputs)
        # Must be either move right or move down from (0,0)
        self.assertIn(new_pos, [[1, 0], [0, 1]])


if __name__ == '__main__':
    unittest.main()


