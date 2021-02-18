"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
All credit too open-ai's examples: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

import gym
import numpy as np
import retro

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, players, combos):
        assert(isinstance(env, retro.retro_env.RetroEnv))
        assert(isinstance(env.action_space, gym.spaces.MultiBinary))
        assert(isinstance(players, int))
        assert(isinstance(combos, list) or isinstance(combos, tuple))
        
        self.players = players
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        self._combos = combos
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, actionList):
        assert(isinstance(actionList, (list, tuple, np.ndarray))) 
        assert(len(actionList) == self.players)
        
        inputs = []
        for action in actionList:
            inputs += list(self._decode_discrete_action[action].copy())
        
        return inputs

    def reward(self, reward):
        if isinstance(reward, int): reward = [reward]
        if self.players == 2: reward[1] = -reward[0]
        return reward

    def get_action_meaning(self, actionList):
        assert(isinstance(actionList, (list, tuple, np.ndarray)))
        assert(len(actionList) == self.players)

        meanings = [self._combos[action] for action in actionList]
        return meanings

class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env, players):
        super().__init__(env=env, players=players, combos=[[], 
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

"""
    Initializes an example discrete environment and randomly selects moves for the agent to make.
    The meaning of each selected move in terms of what buttons are being pressed is also displayed.
"""
def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = StreetFighter2Discretizer(env, env.players)
    env.reset()
    while True:
        env.render()
        actionList = [env.action_space.sample() for i in range(env.players)]
        _, _, _, _ = env.step(actionList)
        print(env.get_action_meaning(actionList))

    env.close()


if __name__ == '__main__':
    main()
