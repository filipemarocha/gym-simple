import unittest

import gym
import gym_simple


class PutBallInBoxEnvTestCase(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('PutBallInBoxEnvFixed8x8-v0')

    def test_update_state_after_pick_up(self):
        self.env.current_state = [(6, 3), (6, 3), (3, 4), 0]
        self.env._update_state_after_pick_up()
        self.assertEqual(self.env.current_state, [(6, 3), (6, 3), (3, 4), 1])

    def test_update_state_after_move(self):
        self.env.current_state = [(6, 3), (6, 3), (3, 4), 0]
        self.env._update_state_after_move(0)
        self.assertEqual(self.env.current_state, [(5, 3), (6, 3), (3, 4), 0])
        self.env._update_state_after_move(1)
        self.assertEqual(self.env.current_state, [(6, 3), (6, 3), (3, 4), 0])

    def test_update_state_after_put_down(self):
        self.env.current_state = [(6, 3), (6, 1), (6, 3), 2]
        self.env._update_state_after_put_down()
        self.assertEqual(self.env.current_state, [(6, 3), (6, 1), (6, 3), 0])

    def test_step_to_goal(self):
        self.env.current_state = [(6, 3), (6, 3), (6, 3), 1]
        goal_state = self.env.step(9)
        self.assertListEqual(list(goal_state[0]), [6, 3, 6, 3, 6, 3, 0])
        self.assertEqual(goal_state[1], 100.0)
        self.assertEqual(goal_state[2], True)

    def test_step_not_to_goal(self):
        self.env.current_state = [(6, 3), (6, 3), (6, 3), 1]
        not_goal_state = self.env.step(8)
        self.assertListEqual(list(not_goal_state[0]), [6, 3, 6, 3, 6, 3, 1])
        self.assertEqual(not_goal_state[1], -1.0)
        self.assertEqual(not_goal_state[2], False)

    def test__reset(self):
        old_initial_state = self.env.initial_state
        self.env._reset()
        self.assertEqual(old_initial_state, self.env.initial_state)
        self.env.fixed_initial_state = False
        old_initial_state = self.env.initial_state
        new_initial_state = self.env.reset()
        self.assertNotEqual(old_initial_state, new_initial_state)


if __name__ == '__main__':
    unittest.main()
