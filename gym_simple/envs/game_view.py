import os, sys
import pygame
import numpy as np

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dsp'

snd_func_card_driver
WHITE_RGB = 255, 255, 255

class GameView:

    def __init__(self, space_size):
        pygame.init()
        self.space_size = space_size
        self.screen = None
        self.screen_size = space_size[0]*120, space_size[1]*120
        dir_name = os.path.join(os.path.split(os.path.abspath(__file__))[0],
            "../images")
        self.agent = pygame.image.load(os.path.join(dir_name,
            "agent_transparent_small.png"))
        self.agentrect = self.agent.get_rect()
        self.ball = pygame.image.load(os.path.join(dir_name,
            "ball_transparent_small.png"))
        self.ballrect = self.ball.get_rect()
        self.box = pygame.image.load(os.path.join(dir_name,
            "box_transparent_small.png"))
        self.boxrect = self.box.get_rect()

    def draw_state(self, state):
        """
         State descrition:
         Agent position (x,y)
         Ball position (x,y)
         Box position (x,y)
         Agent holding: 0-Nothing, 1-Ball, 2-Box
         state is a list: [agent_coordinates, ball_coordinates, box_coordinates,
            agent_is_holding]
        """
        agent_grid_pos = state[0]
        ball_grid_pos = state[1]
        box_grid_pos = state[2]
        agent_holding = state[3]

        # Box and ball in floor relative positions to top left cell coordinates
        box_rel_pos = (40, 96)
        ball_rel_pos = (90, 102)
        if agent_holding == 1:
            ball_rel_pos = (90, 86)
        elif agent_holding == 2:
            box_rel_pos = (40, 80)

        self.screen.fill(WHITE_RGB)
        self.draw_grid_lines()
        self.agentrect.topleft = (agent_grid_pos[0] * 120,
            agent_grid_pos[1] * 120)
        self.screen.blit(self.agent, self.agentrect)
        self.ballrect.center = (ball_rel_pos[0] + ball_grid_pos[0] * 120,
            ball_rel_pos[1] + ball_grid_pos[1] * 120)
        self.screen.blit(self.ball, self.ballrect)
        self.boxrect.center = (box_rel_pos[0] + box_grid_pos[0] * 120,
            box_rel_pos[1] + box_grid_pos[1] * 120)
        self.screen.blit(self.box, self.boxrect)

    def draw_grid_lines(self):
        for row in range(self.space_size[0]):
            for column in range(self.space_size[1]):
                pygame.draw.rect(self.screen, (211, 211, 211),
                    (column * 120, row * 120, 120, 120), 1)

    def update(self, state):
        pygame.init()
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        sys.exit()
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.screen_size)
        self.draw_state(state)
        pygame.display.update()
        return np.flipud(np.rot90(pygame.surfarray.array3d(
            pygame.display.get_surface())))
