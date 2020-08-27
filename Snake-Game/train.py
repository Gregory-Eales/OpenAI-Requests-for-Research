import torch
import numpy as np
import gym
import time

from agent.ppo import PPO


def train(env, agent, episodes, steps):

	for e in range(episodes):

		state = env.reset()

		for i in range(steps):

			action = agent.act(state)
			prev_state = state
			state, reward, done, info = env.step(action)

			agent.store(state, prev_state, action, reward, done)
			
			if done:
				break

		agent.update()


def main():

	env = gym.make('gym_snake:snake-v0')
	agent = PPO()
	train(env, agent, episodes=10, steps=200)

if __name__ == "__main__":
	main()

