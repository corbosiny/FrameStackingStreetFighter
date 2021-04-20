import argparse
import gym
import retro
import threading
import os
import numpy
import time
import random
import numbers
from collections import deque

from tensorflow.python import keras
from keras.models import load_model

class Agent():
    """ 
    Abstract class that user created Agents should inherit from.
    Contains helper functions for launching training environments and generating training data sets.
    """

    # Global constants keeping track of some input lag for some directional movements
    # Moves following these inputs will not be picked up unless input after the lag

    # The indices representing what each index in a training point represent
    TRAINING_POINT_SIZE = 7                                                                        # The number of elements in the vector that makes up a training point
    OBSERVATION_INDEX = 0                                                                          # The current display image of the game state
    STATE_INDEX = 1                                                                                # The state the agent was presented with    
    ACTION_INDEX = 2                                                                               # The action the agent took
    REWARD_INDEX = 3                                                                               # The reward the agent received for that action
    NEXT_OBSERVATION_INDEX = 4                                                                     # The current display image of the new state the action led to
    NEXT_STATE_INDEX = 5                                                                           # The next state that the action led to
    DONE_INDEX = 6                                                                                 # A flag signifying if the game is over

    MAX_DATA_LENGTH = 50000                                                                        # Max number of decision frames the Agent can remember from a fight, average is about 2000 per fight

    DEFAULT_MODELS_DIR_PATH = '../local_models'               # Default path to the dir where the trained models are saved for later access
    DEFAULT_MODELS_SUB_DIR = '{0}'                            # Models are further organized into subdirectories to avoid checkpoint overwrites by this naming scheme
    DEFAULT_MODEL_FILE_EXTENSION = '.model'                   # Extension used to identify saved model weight files versus logs
    DEFAULT_LOG_FILE_EXTENSION = '.log'                       # Extension used to identify training logs versus saved model weight files


    ### End of static variables 

    ### Object methods

    def __init__(self, load= False, name= None, character= "ryu", verbose= False):
        """
        Initializes the agent and the underlying neural network
        
        Parameters
        ----------
        load
            A boolean flag that specifies whether to initialize the model from scratch or load in a pretrained model
        
        name
            A string representing the name of the model that will be used when saving the model and the training logs
            Defaults to the class name if none are provided

        character
            String representing the name of the character this Agent plays as

        verbose
            A boolean variable representing whether or not the print statements in the class are turned on
            Error messages however are not turned off

        Returns
        -------
        None
        """
        assert(isinstance(load, bool))
        assert(name is None or isinstance(name, str))
        assert(isinstance(character, str))

        if name is None: self.name = self.__class__.__name__
        else: self.name = name
        self.character = character
        self.numMatchesPlayed = 0
        self.numMatchesWon = 0
        self.verbose = verbose
        self.playerNumber = 0

        if self.__class__.__name__ != "Agent":
            if not load: self.model = self.initializeNetwork()    								# Only invoked in child subclasses, Agent has no network
            elif load: self.loadModel()

    def prepareForNextFight(self, env, playerNumber):
        """
        Clears the memory of the fighter so it can prepare to record the next fight and records what it's action space is
        
        Parameters
        ----------
        env
            the environment that the player will be fighting in, used to grab the action space

        playerNumber
            Integer representing whether the Agent is player 1(0) or player 2(1)

        Returns
        -------
        None
        """
        #assert(isinstance(actionSpace, gym.spaces.discrete.Discrete))
        assert(isinstance(playerNumber, int))
        assert(playerNumber in [0, 1])
 
        self.environment = env
        self.actionSpace = env.action_space
        self.playerNumber = playerNumber
        self.memory = deque(maxlen= Agent.MAX_DATA_LENGTH)                                      # Double ended queue that stores states during the game
        self.numMatchesPlayed += 1

    def getRandomMove(self):
        """
        Returns a random set of button inputs
        
        Parameters
        ----------
        None

        Returns
        -------
        move 
        """                                                 
        return self.actionSpace.sample()                               

    def recordStep(self, step):
        """
        Records the last observation, action, reward and the resultant observation about the environment for later training
        
        Parameters
        ----------
        step
            A tuple containing the following elements:
            observation
                The current display image in the form of a 2D array containing RGB values of each pixel
            state
                The state the Agent was presented with before it took an action.
                A dictionary containing tagged RAM data
            action
                Integer representing the last move from the move list the Agent chose to pick
            reward
                The reward the agent received for taking that action
            nextObservation
                The resultant display image in the form of a 2D array containing RGB values of each pixel
            nextState
                The state that the chosen action led to
            done
                Whether or not the new state marks the completion of the emulation

        Returns
        -------
        None
        """
        assert(isinstance(step, (list, tuple)))
        assert(len(step) == Agent.TRAINING_POINT_SIZE)
        assert(isinstance(step[Agent.OBSERVATION_INDEX], numpy.ndarray))
        assert(isinstance(step[Agent.STATE_INDEX], dict))
        assert(isinstance(step[Agent.ACTION_INDEX], numbers.Number))
        assert(isinstance(step[Agent.REWARD_INDEX], numbers.Number))
        assert(isinstance(step[Agent.NEXT_OBSERVATION_INDEX], numpy.ndarray))
        assert(isinstance(step[Agent.NEXT_STATE_INDEX], dict))
        assert(isinstance(step[Agent.DONE_INDEX], bool))

        # If the match is over and the agent's number of rounds won is 2, than they won the match
        if step[Agent.DONE_INDEX]:
            key = "player{0}_matches_won".format(self.playerNumber + 1)
            if step[Agent.NEXT_STATE_INDEX][key] == 2 or step[Agent.STATE_INDEX][key] == 2: self.numMatchesWon += 1 
            
        self.memory.append(step) # Steps are stored as tuples to avoid unintended changes

    def reviewFight(self):
        """
        The Agent goes over the data collected from it's last fight, prepares it, and then runs through one epoch of training on the data
        """
        data = self.prepareMemoryForTraining(self.memory)
        self.model = self.trainNetwork(data, self.model)   		                           # Only invoked in child subclasses, Agent does not learn
        self.saveModel()

    def saveModel(self, lossUpdate= None):
        """
        Saves the currently trained model in the default naming convention ../local_models/{Class_Name}/{Class_Name}.model
        
        Parameters
        ----------
        lossUpdate
            An integer value representing the mean loss after one training epoch,
            will be logged in this model's training log if supplied

        Returns
        -------
        None
        """
        assert(lossUpdate is None or isinstance(lossUpdate, numbers.Number))

        totalDirPath = os.path.join(Agent.DEFAULT_MODELS_DIR_PATH, Agent.DEFAULT_MODELS_SUB_DIR.format(self.name))

        self.model.save(os.path.join(totalDirPath, self.getModelName()))
        if self.verbose: print('{0} Model successfully saved'.format(self.name))
        
        if lossUpdate is not None:
            try:
                with open(os.path.join(totalDirPath, self.getLogName()), 'a+') as file:
                    file.write(str(lossUpdate))
                    file.write('\n')
                    if self.verbose: print('{0} Loss History Successfully Updated'.format(self.name))
            except Exception as e:
                print('Trouble updating {0} loss history:'.format(self.name), e)
        else:
            if self.verbose: print('{0} Loss History was not updated as there were no losses to report'.format(self.name))

    def loadModel(self):
        """
        Loads in pretrained model object ../local_models/{Class_Name}/{Class_Name}.model
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        totalDirPath = os.path.join(Agent.DEFAULT_MODELS_DIR_PATH, Agent.DEFAULT_MODELS_SUB_DIR.format(self.name))
        try:
            self.model = keras.models.load_model(os.path.join(totalDirPath, self.getModelName()))
            if self.verbose: print('{0} Model successfully loaded'.format(self.name))
        except Exception as e:
            print('Trouble Loading {0} Model:'.format(self.name), e)

    def getModelName(self):
        """Returns the formatted model name for the current model"""
        return  self.name + Agent.DEFAULT_MODEL_FILE_EXTENSION

    def getLogName(self):
        """Returns the formatted log name for the current model"""
        return self.name + Agent.DEFAULT_LOG_FILE_EXTENSION

    ### End of object methods

    ### Abstract methods for the child Agent to implement

    def getMove(self, obs, info):
        """
        Returns a set of button inputs generated by the Agent's network after looking at the current observation
        
        Parameters
        ----------
        obs
            The observation of the current environment, 2D numpy array of pixel values
        info
            An array of information about the current environment, like player health, enemy health, matches won, and matches lost, etc.
            A full list of info can be found in data.json

        Returns
        -------
        move
            Integer representing the move that was selected from the move list
        """
        assert(isinstance(obs, numpy.ndarray))
        assert(isinstance(info, dict))

        if self.__class__.__name__ == "Agent":
            move = self.getRandomMove()
            return move
        else:
            raise NotImplementedError("Implement getMove in the inherited agent")

    def initializeNetwork(self):
        """
        To be implemented in child class, should initialize or load in the Agent's neural network
        
        Parameters
        ----------
        None

        Returns
        -------
        model
            A newly initialized model that the Agent will use when generating moves
        """
        raise NotImplementedError("Implement initializeNetwork in the inherited agent")
    
    def prepareMemoryForTraining(self, memory):
        """
        To be implemented in child class, should prepare the recorded fight sequences into training data
        
        Parameters
        ----------
        memory
            A 2D array where each index is a recording of a state, action, new state, and reward sequence
            See readme for more details

        Returns
        -------
        data
            The prepared training data
        """
        raise NotImplementedError("Implement prepareMemoryForTraining in the inherited agent")

    def trainNetwork(self, data, model):
        """
        To be implemented in child class, Runs through a training epoch reviewing the training data and returns the trained model
        
        Parameters
        ----------
        data
            The training data for the model
        model
            The model for the function to train

        Returns
        -------
        model
            The newly trained model
        """
        raise NotImplementedError("Implement trainNetwork in the inherited agent")

    ### End of Abstract methods

    def getName(self):
        """Getter for the name of the model"""
        return self.name

    def getCharacter(self):
        """Getter for the character name this Agent is playing as"""
        return self.character

    def getNumberOfMatchesPlayed(self):
        """Getter for the number of matches this Agent has played"""
        return self.numMatchesPlayed

    def getNumberOfWins(self):
        """Getter for the number of wins this Agent has"""
        return self.numMatchesWon

    def __repr__(self):
        """What to return if type is called on this class or any child class"""
        return "Agent"

    def __str__(self):
        """What to return if an Agent is used in a print statement"""
        return self.name

"""
Make a test Agent and run it through one training run on single player mode of streetfighter
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes agent parameters.')
    parser.add_argument('-r', '--render', action= 'store_true', help= 'Boolean flag for if the user wants the game environment to render during play')
    args = parser.parse_args()
    import Lobby
    testLobby = Lobby.Lobby(mode= Lobby.Lobby_Modes.SINGLE_PLAYER)
    agent = Agent()
    testLobby.addPlayer(agent)
    testLobby.executeTrainingRun(render= args.render)