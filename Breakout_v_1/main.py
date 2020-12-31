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

def create_model():
    # keras example for breakout
    inputs = layers.Input(shape=(105, 80, 4), dtype=np.float32)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(2, activation="linear")(layer5)
    model = keras.Model(inputs=inputs, outputs=action)
    optimizer=keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)  # faster than rmsprop
    model.compile(optimizer, loss=keras.losses.Huber())  # Huber for stability
    return model


model = create_model()
second_model = create_model()
model.summary()


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
epsilon = 1.0
gamma = 0.75
previous_lives = 5
max_memory_length = 10000
improvement_check = 100
iterations = 0
games_played = 0
update_second_model = 5000
batch_size = 32

env = gym.make("BreakoutDeterministic-v4")  # is better than v0
while True:  # training continues forever
    observation = env.reset()  # reseting env
    state = rgb_to_greyscale(observation)  # converts observation (210, 160, 3), to greydownscaled state, (105, 80)
    env.step(1)  # FIRE triggers ball initialization
    episode_reward = 0  # initializes reward for the new episode which follows
    while True:  # agent still has lives
        #         env.render()
        iterations += 1  # increasing total iterations of game
        if epsilon > np.random.rand(1)[0]:  # a leap of fate, exploration
            action = 2 if np.random.random(1)[0] > 0.5 else 3  # random left or right | 3 is left, 2 is right
        else:  # agent must predict action, exploatation
            four_states = np.array(state_history[-4:])  # takes last 4 known states
            four_states = four_states.reshape(105, 80, 4)  # reshapes them into input shape
            predictions = model.predict(np.array([four_states]), verbose=0)  # gets reward predictions for both actions
            action = 2 if predictions[0, 0] > predictions[
                0, 1] else 3  # choses the actions with the greatest predicted reward

        if epsilon > 0.05:  # decay over time is applied to epsilon until it reaches critical value
            epsilon -= epsilon / 10000 * np.random.random(1)   # decrease is done by (at least) 0.01 %, critical value is reached in (at least) 29956 steps

        observation, reward, done, info = env.step(
            action)  # action is played, returns new observation, possible reward, done flag and lives remained
        next_state = rgb_to_greyscale(
            observation)  # converts observation (210, 160, 3), to greydownscaled state, (105, 80)  # TODO: check if next_state is really needed, we might only use state

        if info["ale.lives"] != previous_lives:  # if number of lives decreased during this frame
            env.step(1)  # FIRE resummons ball
            previous_lives = info["ale.lives"]  # updates previous_lives with current lives
            if done:
                reward -= 10  # updates reward with negative value because a life was lost

        # uncomment this later
        # if reward == 0:  # if no reward is received
        #     reward -= 0.1  # reward receives small negative value, should encourage the agent to finish the game faster

        # saving values
        state_history.append(state)
        action_history.append(action)
        reward_history.append(reward)
        next_state_history.append(
            next_state)  # next_state of state_history[3] = state_history[4]  # TODO: only use state_history
        state = next_state  # replaces old state with new one

        episode_reward += reward  # increases reward for this episode, for checking out improvements for games

        # TODO: apply backprop sometimes in the future

        # Start Back Prop

        if iterations % batch_size == 0:  # doing backprop once every 32 steps
            indices = np.random.choice(range(4, len(action_history)),
                                       size=batch_size)  # get only indices that have at least 4 previous states, and 1 next state

            state_sample = np.array([state_history[i - 4:i] for i in
                                     indices])  # takes groups of 4 images of game board, previous and except current index
            state_sample = state_sample.reshape(batch_size, 105, 80,
                                                4)  # reshapes group from (32, 4, 105, 80) to (32, 105, 80, 4)
            next_state_sample = np.array([state_history[i - 3: i + 1] for i in
                                          indices])  # takes gropus of 4 images of game board, previous and including current index
            next_state_sample = next_state_sample.reshape(batch_size, 105, 80, 4)
            reward_sample = np.array([reward_history[i] for i in indices])  # has shape (32,)
            action_sample = [action_history[i] - 2 for i in
                             indices]  # has len 32; 2 is decreased from each action to transform it into 0 or 1, to minimize one_hot_vectors size
            future_rewards = np.amax(second_model.predict(next_state_sample, verbose=0),
                                     axis=1)  # gets maximum prediction using second model of future rewards for each next state sample
            updated_q_values = reward_sample + gamma * future_rewards  # for current state, adds reward obtained to next state predicted max reward
            masks = to_categorical(
                action_sample)  # one hot masks are created for actions, to apply backprop only for chosen actions

            with tf.GradientTape() as tape:  # Copied example from keras q-learning. Applies backpropagation to model
                q_values = model(
                    state_sample)  # same as `q_values = model.predict(state_sample, verbose=0)`, but returns tensor
                q_action = tf.reduce_sum(tf.multiply(q_values, masks),
                                         axis=1)  # same as `q_action = np.sum(q_values * masks, axis=1)`, but returns tensor
                loss = loss_function(updated_q_values,
                                     q_action)  # calculates the loss between updated_q_values, which are correct labels expected, and q_action is the output obtained
                grads = tape.gradient(loss, model.trainable_variables)  # yess, applies gradient to weights
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))  # yess, uses optimizer to update wigths

        # End Back Prop

        if iterations % update_second_model == 0:  # once every 5000 iterations
            second_model.set_weights(model.get_weights())  # updates second model

        if len(action_history) > max_memory_length:  # if max memory was reached
            del state_history[:5000]  # deletes first 5000 elements from each list
            del action_history[:5000]
            del reward_history[:5000]
            del next_state_history[:5000]

        if done:  # end game flag
            games_played += 1  # increasing played games
            if games_played % improvement_check == 0:  # once every 100 played games
                model.save("a.h5")
                print(f"Reward: {episode_reward}, games played: {games_played}, iterations made: {iterations}")
            break  # exits current game

    if iterations % 10000 == 0:
        print(f"Reward: {episode_reward}, games played: {games_played}, iterations made: {iterations}")
        print(games_played)
# env.close()
