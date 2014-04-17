# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
This file contains all of the agents that can be selected to
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import sys
import copy

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.  In this phase, the agent should compute the path to the
        goal and store it in a local variable.  All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        bottom, left = 1, 1
        self.corners = ((left, bottom), (left,top), (right, bottom), (right, top))
        
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded

        self.startState = (self.startingPosition, self.corners)

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        return self.startState

    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        return len(state[1]) == 0

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            x,y = state[0][0], state[0][1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            cornersLeft = state[1]
            if (nextx, nexty) in cornersLeft:
                for i in range(len(cornersLeft)):
                    if cornersLeft[i] == (nextx, nexty):
                        break
                cornersLeft = cornersLeft[:i] + cornersLeft[i+1:]
            if not hitsWall:
                nextState = ((nextx, nexty), cornersLeft)
                cost = 1
                successors.append( (nextState, action, cost) )

        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible (as well as consistent).
    """
    
    distance = 0
    cornersLeft = list(state[1])
    pacman = state[0]
    closest = 0
    cornersList = []
    minI = 0
    distance_to_nearest = 0
    total = 0
    
    if len(cornersLeft) > 0:
        for i in range(len(cornersLeft)):
            corner = cornersLeft[i]
            cornersList.append(abs(pacman[0] - corner[0]) + abs(pacman[1] - corner[1]))
        distance_to_nearest = min(cornersList)      
        minI = cornersList.index(distance_to_nearest)
        closest = cornersLeft[minI]
      
        cornersLeft.remove(closest)
        while len(cornersLeft) > 0:
          distanceList = []
          xy1 = closest
          for i in range(len(cornersLeft)):
              xy2 = cornersLeft[i]
              distanceList.append(abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1]))
          closest2 = min(distanceList)
          minI = distanceList.index(closest2)
          closest = cornersLeft[minI]
          cornersLeft.remove(closest)
          
          total = total + closest2
        distance = distance_to_nearest + total
      
    return distance

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start
    
    def getStartStateFull(self):
        return self.startingGameState

    def getHeuristicInfo(self):
        return self.heuristicInfo

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def distanceToAll(width, height, foods, walls):
    print walls
    s = ""
    for w in xrange (width):
        for h in xrange (height):
            p = (w,h)
            if p not in walls:
                s += " "
            else:
                s += "X"
        s += "\n"
    print s


    return {}

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    walls = problem.walls.asList()
    length = len(walls)
    highWall = list(walls[length-1])
    halfBoard = (highWall[0]+1)/2
    foodListTwo=[]
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    
    
    elements = False
    foodList = state[1].asList()
    entireList = len(foodList)
    xy = list(state[0])
      
    lowest = sys.maxint
    largest = -1
    for i in foodList:
      counter = 0
      counter = counter + 1
      if i[0] > halfBoard:
        counter == 0
        foodListTwo.append(i)  
    
    if len(foodListTwo) != 0:
      elements = True
    if elements and counter == 0:
    
      for i in foodListTwo:
        if state[0] != i:
          testing = abs(xy[0] - i[0]) + abs(xy[1] - i[1])
        
          if(testing < lowest):
            lowest = testing
        else:
          lowest = 0
    elif elements:
      lowest = -1
      a1 = state[0]
      a2 = state[0]
      for i in foodListTwo:
        if state[0] != i:
          testing = abs(xy[0] - i[0]) + abs(xy[1] - i[1])
        
          if(testing > lowest):
            lowest = testing
            a2 = i
      if lowest < 0:
        lowest = 0
      elif lowest > (foodGrid.width+foodGrid.height)/2-1:
        if lowest + (foodGrid.width+foodGrid.height)/2 <= foodGrid.width+foodGrid.height:
          hasWall = False
          for x in xrange(min(a1[0], a2[0]), max(a1[0],a2[0])):
            for y in xrange(min(a1[1], a2[1]), max(a1[1],a2[1])):
              if (x,y) in walls:
                hasWall = True
                break
            if hasWall:
              lowest = pow(lowest,len(foodGrid.asList()))
              break

    elif not elements:
    
      isGoalNode = False
   
      xy = list(state[0])
      lowest = sys.maxint
      counter = 0
      
      currentLength = len(foodList)
      
      for i in foodList:
        if state[0] != i:
          
          testing = abs(xy[0] - i[0]) + abs(xy[1] - i[1])
        
          if testing < lowest:
            lowest = testing
        else:
          return 0
    if lowest == sys.maxint:
      lowest = 0
    return lowest


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        return search.aStarSearch(problem)

        
class AnyFoodSearchProblem(PositionSearchProblem):
    """
      A search problem for finding a path to any food.

      This search problem is just like the PositionSearchProblem, but
      has a different goal test, which you need to fill in below.  The
      state space and successor function do not need to be changed.

      The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
      inherits the methods of the PositionSearchProblem.

      You can use this search problem to help you fill in
      the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state
        return self.food[x][y]
        

##################
# Mini-contest 1 #
##################

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    def registerInitialState(self, state):
      "This method is called before any moves are made."
      "*** YOUR CODE HERE ***"
      starttime = time.time()
      self.actions = []
      currentState = state
      
      flag = True
      while(currentState.getFood().count() > 0):
        if flag:
            DFSTuple =self.quickDFS(currentState)
#            US = self.uniformSearch(AnyFoodSearchProblem(currentState))
            nextPathSegment = DFSTuple[0]
#            nextPathSegment = USTuple
            flag = DFSTuple[1]
        else:
            nextPathSegment = self.quickFindPathToDot(currentState) 
            
        self.actions += nextPathSegment
        for action in nextPathSegment: 
          currentState = currentState.generateSuccessor(0, action)
      self.actionIndex = 0
      print('Path found with total cost of %d in %.1f seconds' % (len(self.actions), time.time() - starttime))
      
    def uniformSearch(self, problem):
        explored = {}
        start = problem.getStartState()
        Goal = ""
        explored[start] = ([], 0)
        orderedList = []
        succs = problem.getSuccessors(start)
        for s in succs:
            orderedList.append((s[0], s[1], s[2], start))
        orderedList = sorted(orderedList, key=lambda orderedList:(orderedList[2]))
        while not (orderedList == []):
            curr = orderedList[0]
            pred = curr[3]
            if pred not in explored:
                return []
            if problem.isGoalState(curr[0]):
                Goal = curr[0]
                cost = curr[2]
                if curr[0] not in explored:
                    path = copy.deepcopy(explored[pred][0])
                    path.append(curr[1])
                    explored[curr[0]] = (path, cost)
                    break
            else:
                if curr[0] not in explored:
                    path = copy.deepcopy(explored[pred][0])
                    path.append(curr[1])
                    cost = curr[2]
                    explored[curr[0]] = (path, cost)
                    succs = problem.getSuccessors(curr[0])
                    for s in succs:
                        orderedList.append((s[0], s[1], s[2]+cost, curr[0]))
                else:
                    cost = curr[2]
                    if explored[curr[0]][1] > cost:
                        path = copy.deepcopy(explored[pred][0])
                        path.append(curr[1])
                        explored[curr[0]] = (path, cost)

            orderedList = orderedList[1:]
            orderedList = sorted(orderedList, key=lambda orderedList:(orderedList[2]))
        #      print orderedList
        return explored[Goal][0]

    def quickDFS(self,gameState):
        startPosition = gameState.getPacmanPosition()
        foodGrid = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        
        successorStack = util.Stack()
        finalDirectionArray = []
        visitedNodeArray = []
        previousNodeDict = {}
        directionDict = {}
        goalState = []
        finalNodeArray = []
        visitedNodeArray.append(startPosition)
        for item in problem.getSuccessors(startPosition):
            successorStack.push(item)
        flag = False
        direct = []
        while not successorStack.isEmpty():
            tempSuccessor = successorStack.pop()
            if(problem.isGoalState(tempSuccessor[0])):
                direct.append(tempSuccessor[1])
                flag = True
                break
        return (direct,flag)
      
    def quickFindPathToDot(self,gameState):
        startPosition = gameState.getPacmanPosition()
        foodGrid = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
        successorPriorityQueue = util.PriorityQueue()
        finalDirectionArray = []
        visitedNodeArray = []
        previousNodeDict = {}
        directionDict = {}
        goalState = []
        finalNodeArray = []
        visitedNodeArray.append(startPosition)
    
        for item in problem.getSuccessors(startPosition):
            previousNodeDict[item[0]] = startPosition
            directionDict[item[0]] = item[1]
            successorPriorityQueue.push(item,problem.getCostOfActions([item[1]])+self.quickHeuristic(foodGrid, startPosition))
        #begin A-STAR Search
        while not successorPriorityQueue.isEmpty():
            tempSuccessor = successorPriorityQueue.pop()
            visitedNodeArray.append(tempSuccessor[0])
            if(problem.isGoalState(tempSuccessor[0])):
                goalState = tempSuccessor[0]
                break
            for item in problem.getSuccessors(tempSuccessor[0]):
                if item[0] not in visitedNodeArray and item[0] not in previousNodeDict.keys():
                    previousNodeDict[item[0]] = tempSuccessor[0]
                    directionDict[item[0]] = item[1]
                    nodeToStartArray = []
                    directionToStartArray = []
                    iter = item[0]
                    while True:
                        nodeToStartArray.insert(0,iter)
                        if previousNodeDict[iter] != startPosition:
                            iter = previousNodeDict[iter]
                        else:
                            break
                    for eachNode in nodeToStartArray:
                        directionToStartArray.append(directionDict[eachNode])
                    successorPriorityQueue.push(item,problem.getCostOfActions(directionToStartArray)+self.quickHeuristic(foodGrid, startPosition))
        while True:
            finalNodeArray.insert(0, goalState)
            if previousNodeDict[goalState] != problem.getStartState():
                goalState = previousNodeDict[goalState]
            else:       
                break
        for item in finalNodeArray:
            finalDirectionArray.append(directionDict[item])
        return finalDirectionArray

        
    def quickHeuristic(self,foodGrid,position):
        foodList = foodGrid.asList()
        distance = 0.0

        if(len(foodList) == 0):
            return distance
    
        for food in foodList:
            tmpDistance = ( (position[0] - food[0]) ** 2 + (position[1] - food[1]) ** 2 ) ** 0.5

            if tmpDistance > distance:
                distance = tmpDistance
     
        return distance
    
    def getAction(self, state):
      """
      From game.py: 
      The Agent will receive a GameState and must return an action from 
      Directions.{North, South, East, West, Stop}
      """ 
      "*** YOUR CODE HERE ***"

      if 'actionIndex' not in dir(self): self.actionIndex = 0
      i = self.actionIndex
      self.actionIndex += 1
      if i < len(self.actions):
        return self.actions[i]    
      else:
        return Directions.STOP

    

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(search.bfs(prob))