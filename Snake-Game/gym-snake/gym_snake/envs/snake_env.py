import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
from pygame.locals import *

import numpy as np

class SnakeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, size=16):

      self.size = size
      self.reset()
      pygame.init()

      self.screen = pygame.display.set_mode([800, 800])
      self.screen.fill((0, 0, 0))

  def step(self, action):
    
      state = self.take_action(action)
      reward = self.get_reward()
      done = self.is_terminal()

      return state, reward, done, {}

  def reset(self):

      self.state = np.zeros([self.size, self.size])
      self.player = [[self.size//2, self.size//2]]


  def render(self, mode='human'):
    
      if mode == 'human':
        
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
              self.terminal = True

        self.draw_player()
        self.draw_apple()

        pygame.display.flip()

  def close(self):
      pass

  def draw_player(self):

    for coord in self.player:
      pygame.draw.rect(self.screen, (255, 255, 255), coord+[20, 20])

  def draw_apple(self):
    pass

  def take_action(self, action):

    # 
    if action == 0:
      pass

    if action == 1:
      pass

    if action == 2:
      pass

    if action == 3:
      pass






def main():

    env = SnakeEnv()

    for i in range(10000):
      env.render(mode='human')


if __name__ == "__main__":
  
  main()