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
    """
    A class that handles all of the necessary book keeping for running the gym environment.
    A number of players are added and a game state is selected and the lobby will handle
    piping in the player moves and keeping track of some relevant game information.
    """

    ### Static Variables 

    NO_ACTION = 0                                                                                                   # NO_ACTION is submited when the simulation starts in order to get the initial observation and state info

    DISCRETIZERS = {'StreetFighterIISpecialChampionEdition-Genesis' : StreetFighter2Discretizer}                    # Dictionary of the supported discretized games for this lobby

    STATE_FILE_HEADERS = {Lobby_Modes.SINGLE_PLAYER : "single_player", Lobby_Modes.TWO_PLAYER : "two_player"}       # Lobby can be opened in single player or two player mode and will filter playable states accordingly

    ROUND_TIMER_NOT_STARTED = {'StreetFighterIISpecialChampionEdition-Genesis' : 39208}                             # Stores the starting round timer for each ported game so the lobby can tell when to start recording steps

    ### End of Static Variables

    def __init__(self, game= 'StreetFighterIISpecialChampionEdition-Genesis', mode= Lobby_Modes.SINGLE_PLAYER):
        """
        Initializes the agent and the underlying neural network

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
        assert(isinstance(game, str))
        assert(isinstance(mode, Lobby_Modes))

        self.game = game
        self.mode = mode
        self.clearLobby()

    def getSaveStateList(self):
        """
        Returns a list of all the save state names that can be loaded.
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
        files = os.listdir('../{0}'.format(self.game))
        states = [file.split('.')[0] for file in files if file.split('.')[1] == 'state' and Lobby.STATE_FILE_HEADERS[self.mode] in file]
        stateCharacterNames = [state.split('_')[-1].split('VS') for state in states]
        lobbyCharacterNames = [self.players[playerIndex].character for playerIndex in range(self.mode.value)]
        states = [state for i, state in enumerate(states) if stateCharacterNames[i][0:self.mode.value] == lobbyCharacterNames]

        return states

    def initEnvironment(self, state):
        """
        Initializes a game environment that the Agent can play a save state in

        Parameters
        ----------
        state
            A string of the name of the save state to load into the environment

        Returns
        -------
        None
        """
        assert(isinstance(state, str))
        assert(os.path.exists(os.path.join('../{0}'.format(self.game), state + '.state')))

        self.environment = retro.make(game= self.game, state= state, players= self.mode.value)
        if self.game in Lobby.DISCRETIZERS: self.environment = Lobby.DISCRETIZERS[self.game](self.environment)
        self.environment.reset()                
        # The initial observation and state info are gathered by doing nothing the first frame and viewing the return data       
        self.done = False                                        
        self.lastObservation, _, _, self.lastInfo = self.environment.step([Lobby.NO_ACTION] * self.mode.value)                   

    def addPlayer(self, newPlayer):
        """
        Adds a new player to the player list of active players in this lobby
        will throw a Lobby_Full_Exception if the lobby is full

        Parameters
        ----------
        newPlayer
            An agent object that will be added to the lobby and moves will be requested from when the lobby starts

        Returns
        -------
        None
        """
        assert(newPlayer.__repr__() == "Agent")

        for playerNum, player in enumerate(self.players):
            if player is None:
                self.players[playerNum] = newPlayer
                return

        raise Lobby_Full_Exception("Lobby has already reached the maximum number of players")

    def clearLobby(self):
        """
        Clears the players currently inside the lobby's play queue

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.players = [None] * self.mode.value

    def play(self, state, render= False):
        """
        The Agent will load the specified save state and play through it until finished, recording the fight for training

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
        assert(isinstance(state, str))
        assert(os.path.exists(os.path.join('../{0}'.format(self.game), state + '.state')))
        assert(isinstance(render, bool))

        self.initEnvironment(state)
        [self.players[playerNum].prepareForNextFight(self.environment.action_space, playerNum) for playerNum in range(self.mode.value)]
        self.waitForActionableState(render)

        while not self.done:
            # Get moves for each player
            self.lastAction = [self.players[playerNum].getMove(self.lastObservation, self.lastInfo) for playerNum in range(self.mode.value)]

            # Excute each players moves and calculate rewards
            obs, self.lastReward, self.done, info = self.environment.step(self.lastAction)
            if render: self.environment.render()

            # Record Results
            print('recording')
            [self.players[playerNum].recordStep((self.lastObservation, self.lastInfo, self.lastAction[playerNum], self.lastReward[playerNum], obs, info, self.done)) for playerNum in range(self.mode.value)]
            self.lastObservation, self.lastInfo = [obs, info]                   # Overwrite after recording step so Agent remembers the previous state that led to this one

            # If the round is over wait until the next round starts to fight
            self.waitForActionableState(render)

        # Clean up Environment after the match is over
        self.environment.close()
        if render: self.environment.viewer.close()

    def waitForActionableState(self, render= False):
        """
        Waits to start recording training points again until the game is ready
        This triggers between rounds or before the match has started
        ----------
        render
            Dictionary of the current RAM variables being wathced, keyworded values can be found in Data.json

        Returns
        -------
        None
        """
        assert(isinstance(render, bool))
        while not self.done and not self.isActionableState(self.lastInfo):
            self.lastObservation, _, self.done, self.lastInfo = self.environment.step([Lobby.NO_ACTION] * self.mode.value)
            if render: self.environment.render()


    def isActionableState(self, info):
        """
        Determines if the Agent has control over the game in it's current state(the Agent is in hit stun, ending lag, etc.)
        Parameters
        ----------
        info
            Dictionary of the current RAM variables being wathced, keyworded values can be found in Data.json

        Returns
        -------
        isActionable
            A boolean variable describing whether the Agent has control over the given state of the game
        """
        if info['round_timer'] == Lobby.ROUND_TIMER_NOT_STARTED[self.game]:                                                       
            return False
        elif info['player1_health'] < 0 or info['player2_health'] < 0:
            return False
        else:
            return True

    def executeTrainingRun(self, states= None, review= True, episodes= 1, render= False):
        """
        The lobby will load each of the saved states to generate data for the agent to train on
        Note: This will only work for single player mode

        Parameters
        ----------
        state
            If the user only wants specific states to be trained on, the name of that state can be set here
            If not set the training run will go over all the states made for the current lobby mode
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
        assert(states is None or isinstance(states, (list, tuple)))
        if isinstance(states, (list, tuple)): assert(len(states) != 0)
        assert(isinstance(review, bool))
        assert(isinstance(episodes, int))
        assert(isinstance(render, bool))

        if states is None:                                                      # If no specific states are entered gather all the states of the lobby mode to train on 
            states = self.getSaveStateList()

        for episodeNumber in range(episodes):
            print('Starting episode', episodeNumber)
            for state in states:
                print('Loading {0}..'.format(state))
                self.play(state= state, render= render)
            
                for player in self.players:
                    if player.__class__.__name__ != "Agent" and review == True: 
                        player.reviewFight()

            [print("Player {0} won {1} matches: ".format(i + 1, self.players[i].numMatchesWon)) for i in range(self.mode.value)]


"""
Make a test Agent and run it through one training run on single player mode of streetfighter
"""
if __name__ == "__main__":
    testLobby = Lobby(mode= Lobby_Modes.TWO_PLAYER)
    agent = Agent()
    agent2 = Agent(character="dhalism")
    testLobby.addPlayer(agent)
    testLobby.addPlayer(agent2)
    testLobby.executeTrainingRun(render= True)