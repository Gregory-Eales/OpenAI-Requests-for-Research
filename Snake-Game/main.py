import torch
import numpy as np
import gym
import time

from agent.ppo import PPO


t = time.time()
env = gym.make('gym_snake:snake-v0')

while True:
  action = env.render(mode='human')
  state, reward, done, info = env.step(action)

  if done: break

  time.sleep(0.1)
  print(env.player)

print(state)
print("Took {} seconds".format(round(time.time()-t, 4)))

