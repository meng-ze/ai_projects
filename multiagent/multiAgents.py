# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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

        "Add more of your code here if you want to"

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostdists = []
        fooddists = []
        for ghostState in newGhostStates:
          if ghostState.scaredTimer == 0:
            ghostdists.append(manhattanDistance(newPos, ghostState.getPosition()))

        ghostdists.sort()
        
        for food in newFood.asList():
            fooddists.append(manhattanDistance(newPos, food))

        fooddists.sort()

        if len(fooddists) > 0:
          closestFoodManhattan = fooddists[0]
        else:
          closestFoodManhattan = 0

        numNewFood = successorGameState.getNumFood()

        ghostEvalFunc = 0
        for ghost in newGhostStates:
          ghostdist = manhattanDistance(newPos, ghost.getPosition())
          if ghost.scaredTimer > ghostdist:
            ghostEvalFunc += ghost.scaredTimer - ghostdist

        # if there is a ghost in play that isn't scared, stay away from the nearest one.
        if len(ghostdists) > 0:
          ghostEvalFunc += ghostdists[0]

        return ghostEvalFunc -10*numNewFood - closestFoodManhattan

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

    def maximize(self, gameState, depth, agentIndex):
      maxEval= float("-inf")
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)
        
        # run minimize (the minimize function will stack ghost responses)
        tempEval = self.minimize(successor, depth, 1)
        if tempEval > maxEval:
          maxEval = tempEval
          maxAction = action

      # if this is the first depth, then we're trying to return an ACTION to take. otherwise, we're returning a number. This
      # could theoretically be a tuple with both, but i'm lazy.
      if depth == 1:
        return maxAction
      else:
        return maxEval



    def minimize(self, gameState, depth, agentIndex):
      minEval= float("inf")
      numAgents = gameState.getNumAgents()
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        # if this is the last ghost..
        if agentIndex == numAgents - 1:
          # if we are at our depth limit...
          if depth == self.depth:
            tempEval = self.evaluationFunction(successor)
          else:
            #maximize!
            tempEval = self.maximize(successor, depth+1, 0)
        # we have to minimize with another ghost still.
        else:
          tempEval = self.minimize(successor, depth, agentIndex+1)

        if tempEval < minEval:
          minEval = tempEval
          minAction = action

      return minEval

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
        # maximize legal pacman moves.
        maxAction = self.maximize(gameState, 1, 0)
        return maxAction
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_prune(self, gameState, depth, agentIndex, alpha, beta):
      # init the variables
      maxEval= float("-inf")

      # if this is a leaf node with no more actions, return the evaluation function at this state
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      # otherwise, for evert action, find the successor, and run the minimize function on it. when a value
      # is returned, check to see if it's a new max value (or if it's bigger than the minimizer's best, then prune)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)
        
        # run minimize (the minimize function will stack ghost responses)
        tempEval = self.min_prune(successor, depth, 1, alpha, beta)

        #prune
        if tempEval > beta:
          return tempEval

        if tempEval > maxEval:
          maxEval = tempEval
          maxAction = action

        #reassign alpha
        alpha = max(alpha, maxEval)

      # if this is the first depth, then we're trying to return an ACTION to take. otherwise, we're returning a number. This
      # could theoretically be a tuple with both, but i'm lazy.
      if depth == 1:
        return maxAction
      else:
        return maxEval



    def min_prune(self, gameState, depth, agentIndex, alpha, beta):
      minEval= float("inf")

      # we don't know how many ghosts there are, so we have to run minimize
      # on a general case based off the number of agents
      numAgents = gameState.getNumAgents()

      # if a leaf node, return the eval function!
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      # for every move possible by this ghost
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
      
        # if this is the last ghost to minimize
        if agentIndex == numAgents - 1:
          # if we are at our depth limit, return the eval function
          if depth == self.depth:
            tempEval = self.evaluationFunction(successor)
          else:
            #maximize!
            tempEval = self.max_prune(successor, depth+1, 0, alpha, beta)

        # pass this state on to the next ghost
        else:
          tempEval = self.min_prune(successor, depth, agentIndex+1, alpha, beta)

        #prune
        if tempEval < alpha:
          return tempEval
        if tempEval < minEval:
          minEval = tempEval
          minAction = action

        # new beta
        beta = min(beta, minEval)
      return minEval

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxAction = self.max_prune(gameState, 1, 0, float("-inf"), float("inf"))
        return maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maximize(self, gameState, depth, agentIndex):
      maxEval= float("-inf")
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)


      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)
        
        # run minimize (the minimize function will stack ghost responses)
        tempEval = self.minimize(successor, depth, 1)
        if tempEval > maxEval:
          maxEval = tempEval
          maxAction = action

      # if this is the first depth, then we're trying to return an ACTION to take. otherwise, we're returning a number. This
      # could theoretically be a tuple with both, but i'm lazy.
      if depth == 1:
        return maxAction
      else:
        return maxEval



    def minimize(self, gameState, depth, agentIndex):

      # we will add to this evaluation based on an even weighting of each action.
      minEval= 0
      numAgents = gameState.getNumAgents()
      
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      legalActions = gameState.getLegalActions(agentIndex)
      # calculate the weighting for each minimize action (even distribution over the legal moves).
      prob = 1.0/len(legalActions)
      for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        # if this is the last ghost..
        if agentIndex == numAgents - 1:
          # if we are at our depth limit...
          if depth == self.depth:
            tempEval = self.evaluationFunction(successor)
          else:
            #maximize!
            tempEval = self.maximize(successor, depth+1, 0)
        # we have to minimize with another ghost still.
        else:
          tempEval = self.minimize(successor, depth, agentIndex+1)

        # add the tempEval to the cumulative total, weighting by probability
        minEval += tempEval * prob

      return minEval


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.maximize(gameState, 1, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Our evaluation function begins with the current game score; this helps maintain rewards for going fast and eating ghosts,
      so it is the starting point from which the evaluation function adds and subtracts. By weighting it this way, small things like food pellets
      and time are put into a correct points perspective.

      To encourage pacman to eat food, we subtract 10 points for every food still left in the game. This is a 10 point reward for eating food; just
      like the in game score.

      To encourage pacman to not die by ghost, we find the nearest maze distance [EDIT: MAZEDISTANCE IS OUR BOTTLENECK] to a ghost and add 20 points for every maze distance away the
      ghost is. This utilizes our search algorithms (and the maze distance heuristic) from project 1.

      However, pacman can earn points by eating a ghost (if they are scared), and definitely doesn't need to stay away from them! In this case,
      the heuristic we use to measure a ghost's worth is 20 (scaled for the 200 points you get in game for eating a ghost) minus the distance to
      that ghost (the amount of time it take to reach it)

    """
    # from searchAgents import mazeDistance
    foodMx = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    position = currentGameState.getPacmanPosition()
    foodcount = currentGameState.getNumFood()
    score = currentGameState.getScore()

    nearestGhostDistance = float("inf")

    # evaluate the current state of the ghosts
    ghostEval = 0
    for ghost in ghostStates:
      ghostPosition = (int(ghost.getPosition()[0]), int(ghost.getPosition()[1]))
      md = manhattanDistance(position, ghostPosition)

      if ghost.scaredTimer == 0:
        if md < nearestGhostDistance:
          nearestGhostDistance = md
      #for scared ghosts, evaluate them as 200 points, minus the distance they are away.
      elif ghost.scaredTimer > md:
        ghostEval += 200 - md

    if nearestGhostDistance == float("inf"):
      nearestGhostDistance = 0

    ghostEval += nearestGhostDistance


    return score - 10*foodcount + 1*ghostEval 

# Abbreviation
better = betterEvaluationFunction



# right now our extra credit contest agent simply runs an alpha-beta prune, using the "betterEvalFunction" from q5, with a search depth
# of 4. the score it gets is ~2000...we need an extra bump. I'd like to be able to see the map and ghosts and get a better idea
# what the competition is. It does better when the depth is smaller.. wonder why that is. These ghosts must not be playing to our eval function.
class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    

    def max_prune(self, gameState, depth, agentIndex, alpha, beta):
      # init the variables
      maxEval= float("-inf")

      # if this is a leaf node with no more actions, return the evaluation function at this state
      if len(gameState.getLegalActions(0)) == 0:
        return self.evaluationFunction(gameState)

      # otherwise, for evert action, find the successor, and run the minimize function on it. when a value
      # is returned, check to see if it's a new max value (or if it's bigger than the minimizer's best, then prune)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)
        
        # run minimize (the minimize function will stack ghost responses)
        tempEval = self.min_prune(successor, depth, 1, alpha, beta)

        #prune
        if tempEval > beta:
          return tempEval

        if tempEval > maxEval:
          maxEval = tempEval
          maxAction = action

        #reassign alpha
        alpha = max(alpha, maxEval)

      # if this is the first depth, then we're trying to return an ACTION to take. otherwise, we're returning a number. This
      # could theoretically be a tuple with both, but i'm lazy.
      if depth == 1:
        return maxAction
      else:
        return maxEval



    def min_prune(self, gameState, depth, agentIndex, alpha, beta):
      minEval= float("inf")

      # we don't know how many ghosts there are, so we have to run minimize
      # on a general case based off the number of agents
      numAgents = gameState.getNumAgents()

      # if a leaf node, return the eval function!
      if len(gameState.getLegalActions(agentIndex)) == 0:
        return self.evaluationFunction(gameState)

      # for every move possible by this ghost
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
      
        # if this is the last ghost to minimize
        if agentIndex == numAgents - 1:
          # if we are at our depth limit, return the eval function
          if depth == self.depth:
            tempEval = self.evaluationFunction(successor)
          else:
            #maximize!
            tempEval = self.max_prune(successor, depth+1, 0, alpha, beta)

        # pass this state on to the next ghost
        else:
          tempEval = self.min_prune(successor, depth, agentIndex+1, alpha, beta)

        #prune
        if tempEval < alpha:
          return tempEval
        if tempEval < minEval:
          minEval = tempEval
          minAction = action

        # new beta
        beta = min(beta, minEval)
      return minEval

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.evaluationFunction = ecEvaluationFunction
        self.depth = 4
        maxAction = self.max_prune(gameState, 1, 0, float("-inf"), float("inf"))
        return maxAction

# the evaluation function used in the extra credit.
def ecEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Our evaluation function begins with the current game score; this helps maintain rewards for going fast and eating ghosts,
      so it is the starting point from which the evaluation function adds and subtracts. By weighting it this way, small things like food pellets
      and time are put into a correct points perspective.

      To encourage pacman to eat food, we subtract 10 points for every food still left in the game. This is a 10 point reward for eating food; just
      like the in game score.

      To encourage pacman to not die by ghost, we find the nearest maze distance [EDIT: MAZEDISTANCE IS OUR BOTTLENECK] to a ghost and add 20 points for every maze distance away the
      ghost is. This utilizes our search algorithms (and the maze distance heuristic) from project 1.

      However, pacman can earn points by eating a ghost (if they are scared), and definitely doesn't need to stay away from them! In this case,
      the heuristic we use to measure a ghost's worth is 20 (scaled for the 200 points you get in game for eating a ghost) minus the distance to
      that ghost (the amount of time it take to reach it)

    """
    from searchAgents import mazeDistance
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    position = currentGameState.getPacmanPosition()
    foodcount = currentGameState.getNumFood()
    score = currentGameState.getScore()
    # the amount of food that can be remaining by which point it is not too inefficient to use a mazeDistance heuristic on
    # all foods.

    mazeDistanceFoodCutoff = 20

    nearestGhostDistance = float("inf")

    # evaluate the current state of the ghosts
    ghostEval = 0
    for ghost in ghostStates:
      ghostPosition = (int(ghost.getPosition()[0]), int(ghost.getPosition()[1]))
      md = mazeDistance(position, ghostPosition, currentGameState)

      if ghost.scaredTimer == 0:
        if md < nearestGhostDistance:
          nearestGhostDistance = md
      #for scared ghosts, evaluate them as 200 points, minus the distance they are away.
      elif ghost.scaredTimer > md:
        ghostEval += 200 - md

    if nearestGhostDistance == float("inf"):
      nearestGhostDistance = 0

    ghostEval += nearestGhostDistance

    # find closest food. WE WANT TO BE NEAR THESE! However, mazedistance is inefficient since it runs a 
    # bfs on all food. thus, only late game can we afford to use it.
    closestfood = float("inf")
    if len(foods) < mazeDistanceFoodCutoff:
      for food in foods:
        md = mazeDistance(position, food, currentGameState)
        if md < closestfood:
          closestfood = md
      if closestfood == float("inf"):
        closestfood = 1
    else:
      for food in foods:
        md = manhattanDistance(position, food)
        if md < closestfood:
          closestfood = md
        if closestfood == float("inf"):
          closestfood = 1
    return score - 10*foodcount + 10*ghostEval + 20.0/closestfood 

        
