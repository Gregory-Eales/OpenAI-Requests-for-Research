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
      self.window_size = 800
      self.scale = self.window_size/self.size
      self.reset()
      pygame.init()

      self.screen = pygame.display.set_mode([
        self.window_size,
        self.window_size])

      self.screen.fill((0, 0, 0))

  def step(self, action):
    
      state = self.take_action(action)
      reward = self.get_reward()
      done = self.is_terminal()

      return state, reward, done, {}

  def reset(self):

      self.state = np.zeros([self.size, self.size])
      self.player = [[0, 0]]
      self.player_direction = 1

  def render(self, mode='human'):
      
      r = None

      if mode == 'human':
        
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
              self.terminal = True


          if event.type == KEYDOWN:
            if event.key == K_w:
              r = 0
              print("W pressedd")

            if event.key == K_a:
              r = 1

            if event.key == K_s:
              r = 2

            if event.key == K_d:
              r = 3 

        self.screen.fill((0, 0, 0))
        self.draw_player()
        self.draw_apple()

        pygame.display.flip()

      return r

  def close(self):
      pass

  def draw_player(self):

    for coord in self.player:
      pygame.draw.rect
      (
      self.screen,
      (255, 255, 255),
      [coord[0]*self.scale, coord[1]*self.scale, self.scale, self.scale]
      )

  def draw_apple(self):
    pass

  def update_apple(self): pass

  def update_player(self):

    if self.player_direction == 0:
      
      if self.player[0][1] >= 1:
        self.player[0][1] -= 1

    if self.player_direction == 1:
      if self.player[0][0] > 1:
        self.player[0][0] -= 1

    if self.player_direction == 2:
      if self.player[0][1] < self.size:
        self.player[0][1] += 1

    if self.player_direction == 3:
      if self.player[0][0] < self.size:
        self.player[0][0] += 1

  def take_action(self, action):

    if action != None:
      self.player_direction = action

    self.update_player()
    self.update_apple()

    return self.state

  def get_reward(self):
    return None

  def is_terminal(self): pass

def main():
    import time
    env = SnakeEnv()

    for i in range(50):
      action = env.render(mode='human')
      state, reward, done, info = env.step(action)

      if done: break

      time.sleep(0.1)
      print(env.player)

if __name__ == "__main__":
  
  main()