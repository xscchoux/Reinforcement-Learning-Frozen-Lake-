import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration. Update Q at the end of every episode.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        state = np.random.randint(0,env.nS)
        state_array = [1 if i == state else 0 for i in range(env.nS)]
        env.isd = state_array
        env.reset()
        done = False
        while not done:
            action = epsilon_greedy(Q[state], e)
            next_state, reward, done, prob = env.step(action)
            old_value = Q[state, action]

            next_max = np.max(Q[next_state])
            new_value = (1 - lr) * old_value + lr * (reward + gamma * next_max)
    
            Q[state, action] = new_value
            state = next_state
    # YOUR CODE ENDS HERE
    ############################

    return Q