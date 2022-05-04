# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # loop through the passed number of iterations
        for i in range(self.iterations):

            # loop through all the states in the MDP
            # create temporary dictionary with keys as states and values computed from the current iteration
            temp_dic = util.Counter()
            for s in self.mdp.getStates():

                # find the best action resulting the highest value for the current state
                best_action = self.computeActionFromValues(s)

                # update the current state value
                temp_dic[s] = self.computeQValueFromValues(s, best_action)

            # assign the temporary dictionary to the field self.values
            self.values = temp_dic


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        # look up in the dictionary for the value of the passed state key
        return self.values[state]

    # this function does the summation over the next states
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # initialize the sum over the next states to zero
        rewardSum = 0

        # handel the terminal state with None action
        if action == None:
            # By convention, a terminal state has zero future rewards.
            return 0

        # loop through all the descendant states from state taking passed action
        for index, tuple in enumerate(self.mdp.getTransitionStatesAndProbs(state, action)):
            nextState = tuple[0]
            prob = tuple[1]

            # get the immediate reward for the state, action, nextState transition.
            r = self.mdp.getReward(state, action, nextState)

            # get the term associated with one of the s'
            disc_reward = prob * (r + self.discount * self.getValue(nextState))

            # accumulate the discounted rewards
            rewardSum = rewardSum + disc_reward

        # report the sum over all the s'
        return rewardSum


    # take a state
    # return a tuple (best action, value of the best action) at the passed state
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # handle the terminal state
        if self.mdp.isTerminal(state):
            # return None since there are no legal actions at terminal state
            return None

        # make a dictionary to store Q(s, a) values (the sum over next states) under each legal action at state
        q_table = util.Counter()

        # loop through all the legal actions at state s
        for a in self.mdp.getPossibleActions(state):

            # store the discounted reward associated with each legal action as the key
            q_table[a] = self.computeQValueFromValues(state, a)

        # find the action key which has the highest value and return it
        best_action = q_table.argMax()

        # return the best action
        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

