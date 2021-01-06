import keras
from keras import layers
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import time
import os
import gym
import sys
from gym import error, spaces
from gym import utils
from gym.utils import seeding
try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
            "{}. (HINT: you can install Atari dependencies by running "
            "'pip install gym[atari]'.)".format(e))


def rgb_to_greyscale(observation):
    observation = observation[:,:,0] + observation[:,:,1] + observation[:,:,2]
    return np.where(observation > 0, 255, 0)[::2, ::2]


# Variable declaration
state_history = []
action_history = []
reward_history = []
next_state_history = []
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)  # faster than rmsprop # TODO: somehow set the learning rate higher at start, and decreasing it over time
loss_function = keras.losses.Huber()  # used for stability


# Parameters
epsilon = 0
gamma = 0.5
previous_lives = 5
max_memory_legth = 10000
improvement_check = 100
iterations = 0
games_played = 0
update_second_model = 5000
batch_size = 32

from keras.models import load_model
model = load_model('a.h5')
second_model = load_model('a.h5')

env = gym.make("BreakoutNoFrameskip-v4")  # is better than v0
while True:  # training continues forever
    print(iterations)
    observation = env.reset()  # reseting env
    state = rgb_to_greyscale(observation)  # converts observation (210, 160, 3), to greydownscaled state, (105, 80)

    # when predictiong only LEFT or RIGHT, we have to manually FIRE
    env.step(1)  # FIRE triggers ball initialization

    episode_reward = 0  # initializes reward for the new episode which follows

    while True:  # agent still has lives
        env.render()

        iterations += 1  # increasing total iterations of game
        if epsilon > np.random.rand(1)[0] or iterations < 5:  # a leap of fate, exploration
            # case of model output being only LEFT / RIGHT
            action = 2 if np.random.random(1)[0] > 0.5 else 3  # random left or right | 3 is left, 2 is right

            # case of model output being all 4 actions: NOOP, FIRE, LEFT, RIGHT
        #             action = np.random.choice(4)
        else:  # agent must predict action, exploatation
            four_states = np.array(state_history[-4:])  # takes last 4 known states
            four_states = four_states.reshape(105, 80, 4)  # reshapes them into input shape
            predictions = model.predict(np.array([four_states]), verbose=0)  # gets reward predictions for both actions
            # case of model output being only LEFT / RIGHT
            action = 2 if predictions[0, 0] > predictions[
                0, 1] else 3  # choses the actions with the greatest predicted reward

            # case of model output being all 4 actions: NOOP, FIRE, LEFT, RIGHT
        #             action = np.argmax(predictions[0])

        if epsilon > 0.05:  # decay over time is applied to epsilon until it reaches critical value
            epsilon -= epsilon / 1000000  # critical value is reached in 2995732 steps

        # we should be more interested about total reward later in training, at the beggining we should be greedy for reward
        if gamma < 0.99:
            gamma += gamma / 1000000

        observation, reward, done, info = env.step(
            action)  # action is played, returns new observation, possible reward, done flag and lives remained
        next_state = rgb_to_greyscale(
            observation)  # converts observation (210, 160, 3), to greydownscaled state, (105, 80)

        if info["ale.lives"] != previous_lives:  # if number of lives decreased during this frame
            env.step(1)  # FIRE resummons ball
            previous_lives = info["ale.lives"]  # updates previous_lives with current lives
            # here we tried different negative rewards, including no negative reward for life lost and including no negative reward for DONE
            #             reward -= 1  # updates reward with negative value because a life was lost
            if done:  # if game is finished, agent lost
                reward -= 1

        # we also tried this, the purpose being to encourage the agent to try to make the ball reach above bricks and finish the game faster, but it never reached that state
        # uncomment this later
        # if reward == 0:  # if no reward is received
        #     reward -= 0.1  # reward receives small negative value, should encourage the agent to finish the game faster

        # saving values
        state_history.append(state)
        action_history.append(action)
        reward_history.append(reward)
        state = next_state  # replaces old state with new one

        episode_reward += reward  # increases reward for this episode, for checking out improvements for games

        # Start Back Prop


        if done:  # end game flag
            games_played += 1  # increasing played games
            if games_played % 10 == 0:  # once every 100 played games
                exit()
                model.save("a1.h5")
                print(f"Reward: {episode_reward}, games played: {games_played}, iterations made: {iterations}")
            break  # exits current game

    if iterations % 10000 == 0:
        print(f"Reward: {episode_reward}, games played: {games_played}, iterations made: {iterations}")
        print(games_played)
