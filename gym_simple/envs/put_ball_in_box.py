import copy
import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

from gym_simple.envs.game_view import GameView

class PutBallInBoxEnv(gym.Env):

    def __init__(self, fixed_initial_state, space_size):
        self.current_episode = 0
        self.current_step = 0
        # The same initial state will be used for every episode if
        # fixed initial state is True
        self.fixed_initial_state = fixed_initial_state
        self.space_size = space_size
        # Possible actions of the Agent:
        # Move: up, down, left, right, up-right, up-left, down-right, down-left
        # Pick-up object
        # Put-down object
        self.action_space = spaces.Discrete(10)
        # State descrition:
        # Agent position (x,y)
        # Ball position (x,y)
        # Box position (x,y)
        # Agent holding: 0-Nothing, 1-Ball, 2-Box
        self.observation_space = spaces.Box(low=0, high=1,
            shape=3*space_size+(3,), dtype=np.uint8)
        # To reach the goal: put the ball in the box, the agent needs to pick-up
        # the ball, be in the cell where the box is and put the ball down
        self.ball_in_box = False
        # Initialize state
        self.initial_state = self._get_initial_random_state()
        # So that updates in current state don't affect initial state
        self.current_state = copy.deepcopy(self.initial_state)
        # Initialize Game View
        self.game_view = GameView(space_size)

    def _state_to_vector(self, state):
        return np.array(state[0] + state[1] + state[2] + (state[3],))

    def step(self, action):
        self.current_step += 1
        self._update_state(action)
        if not self._global_state_is_valid(self.current_state):
            raise RuntimeError("State is invalid")
        reward = self._get_reward()
        # If goal is reached, the episode ends
        if self.ball_in_box:
            old_state = self.current_state
            self._reset()
            return self._state_to_vector(old_state), reward, True, {}
        else:
            return self._state_to_vector(self.current_state), reward, False, {}

    def _get_reward(self):
        if self.ball_in_box:
            return 100.0
        else:
            # To punish the longer the time it takes to reach the goal
            return -1.0

    def reset(self):
        self._reset()
        return self._state_to_vector(self.initial_state), 0, False, {}

    def _reset(self):
        """
        Reset the state of the environment and returns an initial state
        """
        self.current_episode += 1
        self.current_step = 0
        self.ball_in_box = False
        # If initial state is not fixed, get a new initial random state
        if not self.fixed_initial_state and self.current_episode > 1:
            self.initial_state = self._get_initial_random_state()
        # So that updates in current state don't affect initial state
        self.current_state = copy.deepcopy(self.initial_state)

    def render(self, mode='human', close=False):
        return self.game_view.update(self.current_state)

    def _get_random_coordinates(self):
        return (np.random.randint(self.space_size[0]),
            np.random.randint(self.space_size[1]))

    def _get_initial_random_state(self):
        # Place randomly one agent, one box and one ball on the floor
        # in a empty 2D space
        agent_coordinates = self._get_random_coordinates()
        ball_coordinates = self._get_random_coordinates()
        box_coordinates = self._get_random_coordinates()
        # Ball and Box cannot be in the same cell on the floor
        # Draw new coordinates until they are not
        while ball_coordinates == box_coordinates:
            box_coordinates = self._get_random_coordinates()
        # Agent starts not holding anything
        agent_is_holding = 0

        return [agent_coordinates, ball_coordinates,
                box_coordinates, agent_is_holding]

    def _update_state(self, action):
        assert action in range(10)
        # If action is Move 0-7
        if action in range(0, 8):
            self._update_state_after_move(action)
        # If action is 8 - Pick up object
        elif action == 8:
            self._update_state_after_pick_up()
        # If action is 9 - Put down object
        elif action == 9:
            self._update_state_after_put_down()

    def _update_state_after_move(self, action):
        old_coordinates = self.current_state[0]
        new_coordinates = self._get_new_coordinates(action)
        # If the coordinates remain the same, return same state
        if new_coordinates != old_coordinates:
            # Change the agents coordinates
            self.current_state[0] = new_coordinates
            # If the agent is holding the ball its coordinates change too
            if self.current_state[3] == 1:
                self.current_state[1] = new_coordinates
            # If the agent is holding the box its coordinates change too
            elif self.current_state[3] == 2:
                self.current_state[2] = new_coordinates

    def _update_state_after_pick_up(self):
        # If the agent is not holding any object he can pick up
        if self.current_state[3] == 0:
            # Agent picks up ball if he is on a cell with a ball on the floor
            if self.current_state[0] == self.current_state[1]:
                self.current_state[3] = 1
            # Agent picks up box if he is on a cell with a box on the floor
            elif self.current_state[0] == self.current_state[2]:
                self.current_state[3] = 2

    def _update_state_after_put_down(self):
        # If agent is not holding an object he can't put down
        if not self.current_state[3] == 0:
            # If agent is holding the ball and both are in the same cell with
            # the box
            if self.current_state[1] == self.current_state[2]:
                # Agent cannot putdown object in a cell already with
                # an object on the floor unless he is holding the ball and puts
                # it in the box, which is the goal
                if self.current_state[3] == 1:
                    self.ball_in_box = True
                    self.current_state[3] = 0
            else:
                # If he is holding something, after put down is not anymore
                self.current_state[3] = 0

    def _coordinates_are_outside(self, coordinates):
        if coordinates[0] in range(0,self.space_size[0]) and  \
            coordinates[1] in range(0,self.space_size[1]):
            return False
        else:
            return True

    def _get_new_coordinates(self, action):
        x_coordinate = self.current_state[0][0]
        y_coordinate = self.current_state[0][1]
        # 0-7 -
        # Move: up, down, left, right, up-right, up-left, down-right, down-left
        if action in [0, 4, 5]:
            # Moves up
            x_coordinate -= 1
        if action in [1, 6, 7]:
            # Moves down
            x_coordinate += 1
        if action in [2, 5, 7]:
            # Moves left
            y_coordinate -= 1
        if action in [3, 4, 6]:
            # Moves right
            y_coordinate +=1
        new_coordinates = (x_coordinate, y_coordinate)
        # Cannot move over the 2D space limits, coordinates remain the same
        # if he tries to cross the limits
        if self._coordinates_are_outside(new_coordinates):
            return self.current_state[0]
        else:
            return new_coordinates

    def _global_state_is_valid(self, global_state):
        # Ball and Box cannot be in the same cell if none is being hold by the
        # agent and ball is not in box
        if global_state[1] == global_state[2] and global_state[3] == 0 \
           and self.ball_in_box == False:
            return False
        # If Agent is holding the ball, both coordinates cannot be different
        if global_state[0] != global_state[1] and global_state[3] == 1:
            return False
        # If Agent is holding the box, both coordinates cannot be different
        if global_state[0] != global_state[2] and global_state[3] == 2:
            return False
        return True


class PutBallInBoxEnvRandom3x3(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvRandom3x3, self).__init__(
            fixed_initial_state = False, space_size=(3, 3)
            )

class PutBallInBoxEnvRandom5x5(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvRandom5x5, self).__init__(
            fixed_initial_state = False, space_size=(5, 5)
            )

class PutBallInBoxEnvRandom8x8(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvRandom8x8, self).__init__(
            fixed_initial_state = False, space_size=(8, 8)
            )

class PutBallInBoxEnvRandom20x20(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvRandom20x20, self).__init__(
            fixed_initial_state = False, space_size=(20, 20)
            )

class PutBallInBoxEnvFixed3x3(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvFixed3x3, self).__init__(
            fixed_initial_state = True, space_size=(3, 3)
            )

class PutBallInBoxEnvFixed5x5(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvFixed5x5, self).__init__(
            fixed_initial_state = True, space_size=(5, 5)
            )

class PutBallInBoxEnvFixed8x8(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvFixed8x8, self).__init__(
            fixed_initial_state = True, space_size=(8, 8)
            )

class PutBallInBoxEnvFixed20x20(PutBallInBoxEnv):

    def __init__(self):
        super(PutBallInBoxEnvFixed20x20, self).__init__(
            fixed_initial_state = True, space_size=(20, 20)
            )
