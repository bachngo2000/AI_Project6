# P5_supplement.py
# -----------
# Reference: this code base is adapted from ai.berkeley.edu, see the license information as follows:
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



######################
# Student Information #
######################
## Write your name and your partner name here


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0  # no noise in the system so the far-sighted agent that see the large +10 reward
    # across the bridge and when it is intended to go east it goes east.
    return answerDiscount, answerNoise

# Prefer the close exit (+1), risking the cliff (-10)
def question3a():
    answerDiscount = 0.5  # makes the agent to have a shorter horizon compared to 0.9
    answerNoise = 0   # no noise decreases the chance of falling off the clif randomly
    answerLivingReward = -1   # punishes the agent for staying on the grid and forces it to move toward exit states
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Prefer the close exit (+1), but avoiding the cliff (-10)
def question3b():
    answerDiscount = 0.5
    answerNoise = 0.1       # just adding 10% of noise in the transition makes it to avoid the clif
    answerLivingReward = -1  # punishes the agent and forces it to move toward exit states
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Prefer the distant exit (+10), risking the cliff (-10)
def question3c():
    answerDiscount = 1   # makes the agent to have a long horizon and see the far terminal state with a high reward
    answerNoise = 0   # no noise decreases the chance of falling off the clif randomly
    answerLivingReward = -1    # punishes the agent and forces it to move toward exit states
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Prefer the distant exit (+10), avoiding the cliff (-10)
def question3d():
    answerDiscount = 1
    # just adding 20% of noise in the transition makes it to avoid the clif
    answerNoise = 0.2   # compared to part b, higher noise is required for the distant exit.
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Avoid both exits and the cliff (so an episode should never terminate)
def question3e():
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = +1  # only rewarding the agent for staying on the grid makes it to avoid the exit states.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import P5_supplement
    for q in [q for q in dir(P5_supplement) if q.startswith('question')]:
        response = getattr(P5_supplement, q)()
        print('  Question %s:\t%s' % (q, str(response)))
