# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from itertools import cycle

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        ghostPositions = successorGameState.getGhostPositions()
        food = currentGameState.getFood()

        ghost_dist = [manhattanDistance(newPos, ghostPos) \
                      for ghostPos in ghostPositions]

        food_dist = [manhattanDistance(newPos, (x, y)) \
                     for x in range(food.width) \
                     for y in range(food.height)
                     if food[x][y] == True]

        food_dist_ratio   = min(food_dist) / float(max(food_dist) - min(food_dist) + 1)
        food_dist_score   = 1 / (food_dist_ratio + 1)
        ghost_dist_score  = min(ghost_dist) / float(min(food_dist) + 1)

        return food_dist_score + ghost_dist_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        #Number of agents
        self.num_agents = gameState.getNumAgents()

        max_util_action = self.max_util(gameState, agentIndex=0)

        return max_util_action[1] # Return max utility

    def getUtility(self, gameState, agentIndex, depth):
        if self.isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState),)

        if agentIndex == 0:
            return self.max_util(gameState, agentIndex, depth)
        else:
            return self.min_util(gameState, agentIndex, depth)

    def max_util(self, gameState, agentIndex, depth=-1):
        v = (float('-inf'), 'STOP')
        depth += 1
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            v_util = self.getUtility(successorState, self.nextAgent(agentIndex), depth)
            if v_util[0] > v[0]:
                v = (v_util[0], action)
        return v

    def min_util(self, gameState, agentIndex, depth):
        v = (float('inf'), 'STOP')
        depth += 1
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            v_util = self.getUtility(successorState, self.nextAgent(agentIndex), depth)
            if v_util[0] < v[0]:
                v = (v_util[0], action)
        return v

    def isTerminalState(self, gameState, depth):
        return len(gameState.getLegalActions()) == 0 \
               or depth == self.depth * self.num_agents - 1

    def nextAgent(self, agentIndex):
        return agentIndex + 1 if agentIndex < (self.num_agents - 1) else 0


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        #Number of agents
        self.num_agents = gameState.getNumAgents()

        alpha = float('-inf')
        beta = float('inf')
        max_util_action = self.max_util(gameState, agentIndex=0, alpha=alpha, beta=beta)

        return max_util_action[1] # Return max utility

    def getUtility(self, gameState, agentIndex, alpha, beta, depth):
        if self.isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState),)

        if agentIndex == 0:
            return self.max_util(gameState, agentIndex, alpha, beta, depth)
        else:
            return self.min_util(gameState, agentIndex, alpha, beta, depth)

    def max_util(self, gameState, agentIndex, alpha, beta, depth=-1):
        v = (float('-inf'), 'STOP')
        depth += 1
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            v_util = self.getUtility(successorState, self.nextAgent(agentIndex), alpha, beta, depth)
            if v_util[0] > v[0]:
                v = (v_util[0], action)
            if v[0] > beta:
                return v
            alpha = max(alpha, v[0])
        return v

    def min_util(self, gameState, agentIndex, alpha, beta, depth):
        v = (float('inf'), 'STOP')
        depth += 1
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            v_util = self.getUtility(successorState, self.nextAgent(agentIndex), alpha, beta, depth)
            if v_util[0] < v[0]:
                v = (v_util[0], action)
            if v[0] < alpha:
                return v
            beta = min(beta, v[0])
        return v

    def isTerminalState(self, gameState, depth):
        return len(gameState.getLegalActions()) == 0 \
               or depth == self.depth * self.num_agents - 1

    def nextAgent(self, agentIndex):
        return agentIndex + 1 if agentIndex < (self.num_agents - 1) else 0


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        #Number of agents
        self.num_agents = gameState.getNumAgents()

        max_util_action = self.max_util(gameState, agentIndex=0)

        return max_util_action[1] # Return max utility

    def getUtility(self, gameState, agentIndex, depth):
        if self.isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState),)

        if agentIndex == 0:
            return self.max_util(gameState, agentIndex, depth)
        else:
            return self.expect_util(gameState, agentIndex, depth)

    def max_util(self, gameState, agentIndex, depth=-1):
        v = (float('-inf'), 'STOP')
        depth += 1
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            v_util = self.getUtility(successorState, self.nextAgent(agentIndex), depth)
            if v_util[0] > v[0]:
                v = (v_util[0], action)
        return v

    def expect_util(self, gameState, agentIndex, depth):
        depth += 1
        v_total = 0  # total utility from all possible actions
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            v_util = self.getUtility(successorState, self.nextAgent(agentIndex), depth)
            v_total += v_util[0]
        v = (v_total / float(len(legalActions)), random.choice(legalActions))
        return v

    def isTerminalState(self, gameState, depth):
        return len(gameState.getLegalActions()) == 0 \
               or depth == self.depth * self.num_agents - 1

    def nextAgent(self, agentIndex):
        return agentIndex + 1 if agentIndex < (self.num_agents - 1) else 0


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    food = currentGameState.getFood()

    ghost_dist = [manhattanDistance(newPos, ghostPos) \
                  for ghostPos in ghostPositions]

    food_dist = [manhattanDistance(newPos, (x, y)) \
                 for x in range(food.width) \
                 for y in range(food.height)
                 if food[x][y] == True]

    food_dist_ratio   = min(food_dist) / float(max(food_dist) - min(food_dist) + 1)
    food_dist_score   = 1 / (food_dist_ratio + 1)
    ghost_dist_score  = min(ghost_dist) / float(min(food_dist) + 1)

    return food_dist_score + ghost_dist_score

# Abbreviation
better = betterEvaluationFunction
