"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
All credit too open-ai's examples: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

import gym
import numpy as np
import retro
import time

class Discretizer(gym.Wrapper):
    """
    Wrap a gym environment and make it use discrete actions and allow for two player matches if the proper state is used.
    """

    ### Static Variables

    FRAME_RATE = 1 / 200                                                                          # The time between frames if rendering is enabled

    ### End of Static Variables 

    def __init__(self, env, combos):
        """
        Initializes the environment wrapper that discretizes the action space and allows for multiple players

        Parameters
        ----------
        env
            The gym environment to wrap around

        combos
            List of filtered discrete actions that make up the action space
            See the StreetFighterWrapper below for an example

        Returns
        -------
        None
        """
        assert(isinstance(env, retro.retro_env.RetroEnv))
        assert(isinstance(env.action_space, gym.spaces.MultiBinary))
        assert(isinstance(combos, (list, tuple)))

        self.players = env.players
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        self._combos = combos
        for combo in combos:
            arr = np.array([False] * int(env.action_space.n / self.players))
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))


    def render(self, mode='human', **kwargs):
        """
        Renders the current contents of the environment and limits it to human watchable speeds

        Parameters
        ----------
        mode 
            string representing the mode to render with
            The render modes are:
                - human: render to the current display or terminal and
                return nothing. Usually for human consumption.

                - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
                representing RGB values for an x-by-y pixel image, suitable
                for turning into a video.
                
                - ansi: Return a string (str) or StringIO.StringIO containing a
                terminal-style text representation. The text can include newlines
                and ANSI escape sequences (e.g. for colors).

        Returns
        -------
        observation
            A 2D numpy array representing the current image buffer data of the environment
        reward
            An array of floats representing the reward for each player
        done
            A boolean representing if the match is over
        info
            A dictionary containing the current metadata extracted from RAM
        """
        returnValues =  self.env.render(mode, **kwargs)
        time.sleep(Discretizer.FRAME_RATE)
        return returnValues

    def step(self, actionList):
        """
        Advances a step in the environment given the selected actions

        Parameters
        ----------
        actionList
            An array of integers where each element is the move selection from one of the players

        Returns
        -------
        observation
            A 2D numpy array representing the current image buffer data of the environment
        reward
            An array of floats representing the reward for each player
        done
            A boolean representing if the match is over
        info
            A dictionary containing the current metadata extracted from RAM
        """
        observation, reward, done, info = self.env.step(self.convertActionListToInputs(actionList))
        return observation, self.calculatePlayerRewards(reward), done, info

    def convertActionListToInputs(self, actionList):
        """
        Converts all of the chosen actions into one input vector to submit to the environment

        Parameters
        ----------
        actionList
            An array of integers where each element is the move selection from one of the players

        Returns
        -------
        inputs
            An array of binary values where each element represents whether a corresponding button on the virtual controllers is being pressed
        """
        assert(isinstance(actionList, (list, tuple, np.ndarray))) 
        assert(len(actionList) == self.players)
        
        inputs = []
        for action in actionList:
            inputs += list(self._decode_discrete_action[action].copy())
        
        return inputs

    def calculatePlayerRewards(self, reward):
        """
        Returns the rewards earned by each player

        Parameters
        ----------
        reward
            Float representing the reward earned by player 1
            Players 2 reward is always the inverse of this

        Returns
        -------
        reward
            An array of of floats containing the rewards for each player
        """
        if isinstance(reward, (int, float)): reward = [reward]
        if self.players == 2: reward[1] = -reward[0]
        return reward

    def get_action_meaning(self, actionList):
        """
        Returns the human description of what button presses represent the chosen actions from each player

        Parameters
        ----------
        actionList
            An array of integers where each element is the move selection from one of the players

        Returns
        -------
        meanings
            A 2D array where each element is a list of all the button presses that represent the corresponding players action    
        """
        assert(isinstance(actionList, (list, tuple, np.ndarray)))
        assert(len(actionList) == self.players)

        meanings = [self._combos[action] for action in actionList]
        return meanings

    def isActionableState(self, info):
        """
        Determines if any players have control over the game in it's current state
        Can be overwrited by a wrapper discretizer for special conditions in other games

        Parameters
        ----------
        info
            Dictionary of the current frame's RAM variables being watched, keyworded values can be found in Data.json

        Returns
        -------
        isActionable
            A boolean variable describing whether the Agent has control over the given state of the game
        """
        raise NotImplementedError("Implement isActionable state in the child class")

    def getCharacterList(self):
        """Getter for the list of characters allowed in this game"""
        return self.characterList

class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        self.ROUND_TIMER_NOT_STARTED = 39208      # Stores the round timer value before countdown has begun so the lobby can tell when to start recording steps
        super().__init__(env=env, combos=[[], 
                                         ['UP'], 
                                         ['DOWN'], 
                                         ['LEFT'], 
                                         ['UP', 'LEFT'],
                                         ['DOWN', 'LEFT'],
                                         ['RIGHT'], 
                                         ['UP', 'RIGHT'], 
                                         ['DOWN', 'RIGHT'],
                                         ['A'],
                                         ['A', 'UP'],
                                         ['A', 'DOWN'],
                                         ['A', 'LEFT'],
                                         ['A', 'RIGHT'],
                                         ['A', 'DOWN', 'LEFT'],
                                         ['A', 'DOWN', 'RIGHT'],
                                         ['B'],
                                         ['B', 'UP'],
                                         ['B', 'DOWN'],
                                         ['B', 'LEFT'],
                                         ['B', 'RIGHT'],
                                         ['B', 'DOWN', 'LEFT'],
                                         ['B', 'DOWN', 'RIGHT'],
                                         ['C'],
                                         ['C', 'UP'],
                                         ['C', 'DOWN'],
                                         ['C', 'LEFT'],
                                         ['C', 'RIGHT'],
                                         ['C', 'DOWN', 'LEFT'],
                                         ['C', 'DOWN', 'RIGHT'],
                                         ['X'],
                                         ['X', 'UP'],
                                         ['X', 'DOWN'],
                                         ['X', 'LEFT'],
                                         ['X', 'RIGHT'],
                                         ['X', 'DOWN', 'LEFT'],
                                         ['X', 'DOWN', 'RIGHT'],
                                         ['Y'],
                                         ['Y', 'UP'],
                                         ['Y', 'DOWN'],
                                         ['Y', 'LEFT'],
                                         ['Y', 'RIGHT'],
                                         ['Y', 'DOWN', 'LEFT'],
                                         ['Y', 'DOWN', 'RIGHT'],
                                         ['Z'],
                                         ['Z', 'UP'],
                                         ['Z', 'DOWN'],
                                         ['Z', 'LEFT'],
                                         ['Z', 'RIGHT'],
                                         ['Z', 'DOWN', 'LEFT'],
                                         ['Z', 'DOWN', 'RIGHT']])


    def isActionableState(self, info):
        if(not hasattr(self, 'prevHealths')): self.prevHealths = [info['player1_health'], info['player2_health']]

        isActionable = True

        if info['round_timer'] == self.ROUND_TIMER_NOT_STARTED:                                                       
            isActionable = False
        elif info['player1_health'] < 0 and info['player2_matches_won'] == 1 and self.prevHealths[0] >= 0: # There is one frame before a player death and the win is counted 
            isActionable = True
        elif info['player2_health'] < 0 and info['player1_matches_won'] == 1 and self.prevHealths[1] >= 0: # There is one frame before a player death and the win is counted
            isActionable = True
        elif info['player1_health'] < 0 or info['player2_health'] < 0:
            isActionable = False
        elif self.prevHealths[0] < 0 and info['player1_health'] == 0:
            isActionable = False
        elif self.prevHealths[1] < 0 and info['player2_health'] == 0:
            isActionable = False
        
        self.prevHealths = [info['player1_health'], info['player2_health']]
        return isActionable

"""
Initializes an example discrete environment and randomly selects moves for the agent to make.
The meaning of each selected move in terms of what buttons are being pressed is also displayed.
"""
def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = StreetFighter2Discretizer(env)
    env.reset()
    while True:
        env.render()
        actionList = [env.action_space.sample() for i in range(env.players)]
        _, _, _, _ = env.step(actionList)
        print(env.get_action_meaning(actionList))

    env.close()


if __name__ == '__main__':
    main()
