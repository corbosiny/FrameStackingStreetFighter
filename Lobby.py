import argparse
import retro
import os
import time
from enum import Enum

from Discretizer import StreetFighter2Discretizer
from Agent import Agent

# Used incase too many players are added to the lobby
class Lobby_Full_Exception(Exception):
    pass

# determines how many players the lobby will request moves from before updating the game state
class Lobby_Modes(Enum):
    SINGLE_PLAYER = 1
    TWO_PLAYER = 2

class Lobby():
    """A class that handles all of the necessary book keeping for running the gym environment.
       A number of players are added and a game state is selected and the lobby will handle
       piping in the player moves and keeping track of some relevant game information.
    """

    ### Static Variables 

    NO_ACTION = 0                                                                        # The Lobby submits the NO_ACTION move until the round actually starts
    ROUND_TIMER_NOT_STARTED = {'StreetFighterIISpecialChampionEdition-Genesis' : 39208}  # This is the value the timer variable in RAM is set to before the round starts
                                                                                         # For each new game you want the lobby to support add in an element for that game's round timer

    FRAME_RATE = 1 / 115                                                                 # The time between frames if rendering is enabled

    ### End of static variables

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', mode= Lobby_Modes.SINGLE_PLAYER):
        """Initializes the agent and the underlying neural network

        Parameters
        ----------
        game
            A String of the game the lobby will be making an environment of, defaults to StreetFighterIISpecialChampionEdition-Genesis

        mode
            An enum type that describes whether this lobby is for single player or two player matches

        Returns
        -------
        None
        """
        assert(type(game) == str)
        assert(type(mode) == Lobby_Modes)

        self.game = game
        self.mode = mode
        self.clearLobby()

    def getSaveStateList(self):
        """Returns a list of all the save state names that can be loaded.
           Assumes there is a directory with the same name as the game
           that contains the save states

        Parameters
        ----------
        None

        Returns
        -------
        states
            A list of strings where each string is the name of a different save state
        """
        files = os.listdir('../{self.game}')
        states = [file.split('.')[0] for file in files if file.split('.')[1] == 'state']
        return states

    def initEnvironment(self, state):
        """Initializes a game environment that the Agent can play a save state in

        Parameters
        ----------
        state
            A string of the name of the save state to load into the environment

        Returns
        -------
        None
        """
        assert(type(state) == str)
        assert(os.path.exists(os.path.join('../{self.game}', state)))

        self.environment = retro.make(game= self.game, state= state, players= self.mode.value)
        self.environment = StreetFighter2Discretizer(self.environment)
        self.environment.reset()                
        # The initial observation and state info are gathered by doing nothing the first frame and viewing the return data                                               
        self.lastObservation, _, _, self.lastInfo = self.environment.step(Lobby.NO_ACTION)                   
        self.done = False
        while not self.isActionableState(self.lastInfo):
            self.lastObservation, _, _, self.lastInfo = self.environment.step(Lobby.NO_ACTION)

    def addPlayer(self, newPlayer):
        """Adds a new player to the player list of active players in this lobby
           will throw a Lobby_Full_Exception if the lobby is full

        Parameters
        ----------
        newPlayer
            An agent object that will be added to the lobby and moves will be requested from when the lobby starts

        Returns
        -------
        None
        """
        assert(type(newPlayer) == Agent)

        for playerNum, player in enumerate(self.players):
            if player is None:
                self.players[playerNum] = newPlayer
                return

        raise Lobby_Full_Exception("Lobby has already reached the maximum number of players")

    def clearLobby(self):
        """Clears the players currently inside the lobby's play queue

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.players = [None] * self.mode.value

    def isActionableState(self, info):
        """Determines if the Agent has control over the game in it's current state(the Agent is in hit stun, ending lag, etc.)

        Parameters
        ----------
        info
            The RAM info of the current game state the Agent is presented with as a dictionary of keyworded values from Data.json

        Returns
        -------
        isActionable
            A boolean variable describing whether the Agent has control over the given state of the game
        """
        assert(type(info) == dict)

        if info['round_timer'] == Lobby.ROUND_TIMER_NOT_STARTED[self.game]:                                                       
            return False
        else:
            return True

    def play(self, state, render= False):
        """The Agent will load the specified save state and play through it until finished, recording the fight for training

        Parameters
        ----------
        state
            A string of the name of the save state the Agent will be playing

        render
            A boolean flag that specifies whether or not to visually render the game while the Agent is playing

        Returns
        -------
        None
        """
        assert(type(state) == str)
        assert(os.path.exists(os.path.join('../{self.game}', state)))
        assert(type(render) == bool)

        self.initEnvironment(state)

        while not self.done:
            # Get moves for each player
            self.lastAction = []
            for playerIndex in range(self.mode.value):
                self.lastAction += self.players[playerIndex].getMove(self.lastObservation, self.lastInfo)

            # Excute each players moves and calculate rewards
            obs, self.lastReward, self.done, info = self.environment.step(self.lastAction)
            if self.mode == Lobby_Modes.TWO_PLAYER: self.lastReward[1] = -self.lastReward[0]    
            if render: 
                self.environment.render()
                time.sleep(Lobby.FRAME_RATE)

            # Record Results
            for playerIndex in range(self.mode.value):
                self.players[playerIndex].recordStep((self.lastObservation, self.lastInfo, self.lastAction[playerIndex], self.lastReward[playerIndex], obs, info, self.done))
                self.lastObservation, self.lastInfo = [obs, info]                   # Overwrite after recording step so Agent remembers the previous state that led to this one
        
        # Clean up Environment after the match is over
        self.environment.close()
        if render: self.environment.viewer.close()

    def executeTrainingRun(self, review= True, episodes= 1, render= False):
        """The lobby will load each of the saved states to generate data for the agent to train on
            Note: This will only work for single player mode

        Parameters
        ----------
        review
            A boolean variable that tells the Agent whether or not it should train after running through all the save states, true means train

        episodes
            An integer that represents the number of game play episodes to go through before training, once through the roster is one episode

        render
            A boolean flag that specifies whether or not to visually render the game while the Agent is playing

        Returns
        -------
        None
        """
        assert(type(review) == bool)
        assert(type(episodes) == int)

        for episodeNumber in range(episodes):
            print('Starting episode', episodeNumber)
            for state in self.getSaveStateList():
                self.play(state= state)
            
            if self.players[0].__class__.__name__ != "Agent" and review == True: 
                self.players[0].reviewFight()


# Makes an example lobby and has a random agent play through an example training run
if __name__ == "__main__":
    testLobby = Lobby()
    agent = Agent()
    testLobby.addPlayer(agent)
    testLobby.executeTrainingRun(render= True)