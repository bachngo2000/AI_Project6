# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        # a dictionary with (state, action) as key and q-value as value
        self.q_dic = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # current Q-value for the state-action.
        # returns 0 for not found keys as initialized by util.Counter()
        # only access Q values by calling getQValue
        return self.q_dic[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # initialize the max q_value to a very large negative number
        max_qvalue = -100000

        # get a list of legal actions at the passed state
        legalActions = self.getLegalActions(state)
        print("legalActions")

        # handle the terminal state where there are no legal actions
        if len(legalActions) == 0:
            # return q-value of 0 since there are no legal actions at terminal state
            return 0

        # find the max q(state,action) for the passed state
        # loop through the dictionary keys for the max value
        for key in self.q_dic: # key is (state, action)
            if key[0] == state and self.getQValue(key[0], key[1]) > max_qvalue:
                # update the max value
                max_qvalue = self.getQValue(key[0], key[1])
        # return the final max value
        return max_qvalue


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # initialize the max q_value to a very large negative number
        max_qvalue = -100000

        # handle the terminal state: there are no legal actions
        legalActions = self.getLegalActions(state)
        print("legalActions", legalActions)
        if len(legalActions) == 0:
            # return None since there are no legal actions at terminal state
            return None

        # find the action for which the q(state,action) is the max for the passed state
        for key in self.q_dic:    # key is (state, action)
            if key[0] == state and self.getQValue(key[0], key[1]) > max_qvalue:
                # update the max value and its key
                max_qvalue = self.getQValue(key[0], key[1])
                max_action = key[1]

        return max_action


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        legalActions = self.getLegalActions(state)
        "*** YOUR CODE HERE ***"
        # handle the terminal state: there are no legal actions
        if len(legalActions) == 0:
            # return None since there are no legal actions at terminal state
            return None

        if random.uniform(0, 1) < self.epsilon:
            # explore action space by taking random actions
            action = random.choice(legalActions)
        else:
            # exploit learned values use the Q-table
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # current Q-value for the state/action couple
        current_value = self.getQValue(state, action)

        # next best Q-value
        next_max_q = self.computeValueFromQValues(nextState)

        # update the Q-value with the Bellman equation using the sample
        new_sample = (reward + self.discount * next_max_q)
        diff = new_sample - current_value
        # update the Q-value
        self.q_dic[(state, action)] = current_value + self.alpha * diff



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
