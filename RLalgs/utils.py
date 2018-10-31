import numpy as np
import random
from time import sleep

def estimate(OldEstimate, StepSize, Target):
    NewEstimate = OldEstimate + StepSize * (Target - OldEstimate)
    return NewEstimate

def epsilon_greedy(value, e, seed = None):
    '''
    Implement Epsilon-Greedy policy
    
    Inputs:
    value: numpy ndarray
            A vector of values of actions to choose from
    e: float
            Epsilon
    seed: None or int
            Assign an integer value to remove the randomness
    
    Outputs:
    action: int
            Index of the chosen action
    '''  
    assert len(value.shape) == 1
    assert 0 <= e <= 1    
    if seed != None:
        np.random.seed(seed)
    ###########################
    # YOUR CODE STARTS HERE
#    k = len(value)
#    N = [0]*k # Initialize play counters:  j = 0 for j = 1,...,k
    if random.random() > e:
        selected_arm = np.argmax(value)
    else:
        selected_arm = random.randrange(len(value))
#    N[selected_arm] += 1
#    action = np.argmax(N)
    action = selected_arm
    # YOUR CODE ENDS HERE
    ###########################
    return action

def action_evaluation(env, gamma, v):
    '''
    
    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount value
    v: numpy ndarray
            Values of states
    Outputs:
    q: numpy ndarray
            Q values of all state-action pairs
    '''
    nS = env.nS
    nA = env.nA
    q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            ############################
            # YOUR CODE STARTS HERE
            temp = 0
            for i in range(len(env.P[s][a])):
                temp += env.P[s][a][i][0]*(env.P[s][a][i][2] + gamma*v[env.P[s][a][i][1]])
            q[s,a] = temp
            # YOUR CODE ENDS HERE
            ############################
    return q

def action_selection(q):
    '''
    Select action from the Q value
    
    Inputs:
    q: numpy ndarray
    
    Outputs:
    actions: int
            The chosen action of each state
    '''
    
    actions = np.argmax(q, axis = 1)    
    return actions 

def render(env, policy):
    '''
    Play games with the given policy
    
    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    policy: numpy ndarray
            Maps state to action
    '''
    
    state = env.reset()
    terminal = False
    
    while not terminal:
        action = policy[state]
        state, reward, terminal, prob = env.step(action)
        env.render()
        sleep(1)
    
    print('Episode ends. Reward =', reward)
    
def human_play(env):
    '''
    Play games
    
    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    '''
    
    print('Action indices: LEFT=0, DOWN = 1, RIGHT = 2, UP = 3')
    state = env.reset()
    env.render()
    terminal = False
    
    while not terminal:
        action = int(input('Give the environment your action index:'))
        state, reward, terminal, prob = env.step(action)
        env.render()